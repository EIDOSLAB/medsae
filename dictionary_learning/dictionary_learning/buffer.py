import torch as t
#from nnsight import LanguageModel
import gc
from medclip import MedCLIPModel, MedCLIPVisionModel
from medclip import MedCLIPProcessor
from functorch import vmap
from tqdm import tqdm
import datetime
import clip


from .config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}


class ActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model , # LanguageModel from which to extract activations
                 submodule, # submodule of the model from which to extract activations
                 d_submodule=None, # submodule dimension; if None, try to detect automatically
                 io='out', # can be 'in' or 'out'; whether to extract input or output activations
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 remove_bos: bool = False,
                 ):
        
        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            try:
                if io == 'in':
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        self.activations = t.empty(0, d_submodule, device=device, dtype=model.dtype)
        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.remove_bos = remove_bos
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.model.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        # Optional progress bar when filling buffer. At larger models / buffer sizes (e.g. gemma-2-2b, 1M tokens on a 4090) this can take a couple minutes.
        # pbar = tqdm(total=self.activation_buffer_size, initial=current_idx, desc="Refreshing activations")

        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                with self.model.trace(
                    self.text_batch(),
                    **tracer_kwargs,
                    invoker_args={"truncation": True, "max_length": self.ctx_len},
                ):
                    if self.io == "in":
                        hidden_states = self.submodule.inputs[0].save()
                    else:
                        hidden_states = self.submodule.output.save()
                    input = self.model.inputs.save()

                    self.submodule.output.stop()
            attn_mask = input.value[1]["attention_mask"]
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            if self.remove_bos:
                hidden_states = hidden_states[:, 1:, :]
                attn_mask = attn_mask[:, 1:]
            hidden_states = hidden_states[attn_mask != 0]

            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            hidden_states = hidden_states[:remaining_space]

            self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
                self.device
            )
            current_idx += len(hidden_states)

            # pbar.update(len(hidden_states))

        # pbar.close()
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'd_submodule' : self.d_submodule,
            'io' : self.io,
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()


class HeadActivationBuffer:
    """
    This is specifically designed for training SAEs for individual attn heads in Llama3. 
    Much redundant code; can eventually be merged to ActivationBuffer.
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model , # LanguageModel from which to extract activations
                 layer, # submodule of the model from which to extract activations
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 apply_W_O = False,
                 remote = False,
                 ):
        
        self.layer = layer
        self.n_heads = model.config.num_attention_heads
        self.resid_dim = model.config.hidden_size 
        self.head_dim = self.resid_dim //self.n_heads
        self.data = data
        self.model = model
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.apply_W_O = apply_W_O
        self.remote = remote

        self.activations = t.empty(0, self.n_heads, self.head_dim, device=device) # [seq-pos, n_layers, n_head, head_dim]
        self.read = t.zeros(0).bool()
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    def refresh(self):
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:
            with t.no_grad():
                with self.model.trace(self.text_batch(), **tracer_kwargs, invoker_args={'truncation': True, 'max_length': self.ctx_len}, remote=self.remote):
                    input = self.model.inputs.save()
                    hidden_states = self.model.model.layers[self.layer].self_attn.o_proj.inputs[0][0]#.save()
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]

                    # Reshape by head
                    new_shape = hidden_states.size()[:-1] + (self.n_heads, self.head_dim) # (batch_size, seq_len, n_heads, head_dim)
                    hidden_states = hidden_states.view(*new_shape)

                    # Optionally map from head dim to resid dim
                    if self.apply_W_O:
                        hidden_states_W_O_shape = hidden_states.size()[:-1] + (self.model.config.hidden_size,) # (batch_size, seq_len, n_heads, resid_dim)
                        hidden_states_W_O = t.zeros(hidden_states_W_O_shape, device=hidden_states.device)
                        for h in range (self.n_heads):
                            start = h*self.head_dim
                            end = (h+1)*self.head_dim
                            hidden_states_W_O[..., h, start:end] = hidden_states[..., h, :]
                        hidden_states = self.model.model.layers[self.layer].self_attn.o_proj(hidden_states_W_O).save()

            # Apply attention mask
            attn_mask = input.value[1]['attention_mask']
            hidden_states = hidden_states[attn_mask != 0]

            # Save results
            self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'layer': self.layer,
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()


class NNsightActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """

    def __init__(
        self,
        data,  # generator which yields text data
        model,  # LanguageModel from which to extract activations
        submodule,  # submodule of the model from which to extract activations
        d_submodule=None,  # submodule dimension; if None, try to detect automatically
        io="out",  # can be 'in' or 'out'; whether to extract input or output activations, "in_and_out" for transcoders
        n_ctxs=3e4,  # approximate number of contexts to store in the buffer
        ctx_len=128,  # length of each context
        refresh_batch_size=512,  # size of batches in which to process the data when adding to buffer
        out_batch_size=8192,  # size of batches in which to yield activations
        device="cpu",  # device on which to store the activations
    ):

        if io not in ["in", "out", "in_and_out"]:
            raise ValueError("io must be either 'in' or 'out' or 'in_and_out'")

        if d_submodule is None:
            try:
                if io == "in":
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError("d_submodule cannot be inferred and must be specified directly")
        
        if io in ["in", "out"]:
            self.activations = t.empty(0, d_submodule, device=device)
        elif io == "in_and_out":
            self.activations = t.empty(0, 2, d_submodule, device=device)

        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]


    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts, return_tensors="pt", max_length=self.ctx_len, padding=True, truncation=True
        )

    def token_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return t.tensor([next(self.data) for _ in range(batch_size)], device=self.device)
        except StopIteration:
            raise StopIteration("End of data stream reached")
        
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        # if batch_size is None:
        #     batch_size = self.refresh_batch_size
        # try:
        #     return [next(self.data) for _ in range(batch_size)]
        # except StopIteration:
        #     raise StopIteration("End of data stream reached")
        return self.token_batch(batch_size)

    def _reshaped_activations(self, hidden_states):
        hidden_states = hidden_states.value
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states = hidden_states.view(batch_size * seq_len, d_model)
        return hidden_states

    def refresh(self):
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:

            with t.no_grad(), self.model.trace(
                self.token_batch(),
                **tracer_kwargs,
                invoker_args={"truncation": True, "max_length": self.ctx_len},
            ):
                if self.io in ["in", "in_and_out"]:
                    hidden_states_in = self.submodule.inputs[0].save()
                if self.io in ["out", "in_and_out"]:
                    hidden_states_out = self.submodule.output.save()

            if self.io == "in":
                hidden_states = self._reshaped_activations(hidden_states_in)
            elif self.io == "out":
                hidden_states = self._reshaped_activations(hidden_states_out)
            elif self.io == "in_and_out":
                hidden_states_in = self._reshaped_activations(hidden_states_in).unsqueeze(1)
                hidden_states_out = self._reshaped_activations(hidden_states_out).unsqueeze(1)
                hidden_states = t.cat([hidden_states_in, hidden_states_out], dim=1)
            self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "io": self.io,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()


import numpy as np
import torchvision
import math


class CLIPActivationBuffer():
    """
    Implements a buffer of activations for CLIP. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is full.
    """
    @classmethod
    def get(cls,
            model_name,
            dataloader, # Dataloader with the same batch size than out_batch_size
            modality, # 'text' or 'img'
            model, # CLIP Model from which to extract activations
            processor, # CLIPProcessor
            layer=None, # submodule of the model from which to extract activations
            out_batch_size=8192, # size of batches in which to yield activations
            device='cpu', # device on which to store the activations
            precompute=False,
            load_activations_path=None, # If you want to load already computed activation
            save_activations_path=None, # If you want to save the computed activations (None : you don't save, str : saving path)
            return_raw_data = False, # If you want to access the raw data corresponding to the activations (e.g. FeatureBuffer)
            shuffle=False # If you want to shuffle the data at each epoch
            ):
        if model_name.startswith("CLIP"):
            return OpenAICLIPActivationBuffer(
                dataloader, 
                modality,
                model,
                processor, 
                layer,
                out_batch_size,
                device,
                precompute,
                load_activations_path,
                save_activations_path,
                return_raw_data,
                shuffle)
        elif model_name.startswith("Med"):
            return MedCLIPActivationBuffer(
                dataloader, 
                modality,
                model,
                processor, 
                layer,
                out_batch_size,
                device,
                precompute,
                load_activations_path,
                save_activations_path,
                return_raw_data,
                shuffle)
        else:
            raise Exception(f"ERROR : {model_name} is not available for this script")
        


    def __init__(self, 
                 dataloader, # Dataloader with the same batch size than out_batch_size
                 modality, # 'text' or 'img'
                 model, # CLIP Model from which to extract activations
                 processor, # CLIPProcessor
                 layer, # submodule of the model from which to extract activations
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 precompute=False,
                 load_activations_path=None, # If you want to load already computed activation
                 save_activations_path=None, # If you want to save the computed activations (None : you don't save, str : saving path)
                 return_raw_data = False, # If you want to access the raw data corresponding to the activations (e.g. FeatureBuffer)
                 shuffle=False, # If you want to shuffle the data at each epoch
                 dtype=t.float32
                 ):
        
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.modality = modality
        model.to(device)
        self.model = model
        self.layer = layer
        if self.layer is not None:
            self.latent_representations = {}
            def get_layer_output(layer, input, output):
                self.latent_representations['act'] = output
            self.layer.register_forward_hook(get_layer_output)  

        self.processor = processor
        self.out_batch_size = out_batch_size
        self.device = device
        self.load_activations = load_activations_path
        self.save_activation_path = save_activations_path
        self.activations_computed = False
        self.return_raw_data = return_raw_data
        self.precompute_idx = 0
        self.activations = None
        self.shuffle = shuffle
        self.dtype = dtype
        if precompute or load_activations_path:
            self.activations = self.precompute_activations()
            
        
    
    def __len__(self):
        if self.activations_computed:
            return math.ceil(len(self.activations)/self.out_batch_size)
        else:  return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        if not(self.activations_computed):    
            self.model.eval()
            with t.no_grad():
                outputs = self.compute_batch()
                if outputs == None:
                    self.refresh()
                    raise StopIteration
                if self.return_raw_data:
                    return outputs[0], outputs[1].to(dtype=self.dtype)
                else:
                    return outputs[1]
        else:
            if self.__len__() > self.precompute_idx:
                outputs = self.activations[self.precompute_idx*self.out_batch_size:(self.precompute_idx+1)*self.out_batch_size]
                self.precompute_idx += 1
                if self.return_raw_data:
                    return self.retrieve_batch_raw()[:len(outputs)], outputs.to(dtype=self.dtype)
                else:
                    return outputs.to(dtype=self.dtype)
            else:
                self.refresh()
                raise StopIteration
    
    def refresh(self):
        self.precompute_idx = 0
        self.activations_computed = True

        if self.shuffle:
            idxs = t.randperm(len(self.activations))
            self.activations = self.activations[idxs]
        
        
        if self.save_activation_path:
            print(f"Saving activations to {self.save_activation_path} ...")
            t.save(self.activations, self.save_activation_path)

    def retrieve_batch_raw(self):
        """
        Return a batch of raw data
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch
    
    def precompute_activations(self):
        if self.load_activations:
            print(f"Loading activations from {self.load_activations}")
            self.activations_computed = True
            activations = t.load(self.load_activations).to(self.device)
            assert len(activations.shape) == 2, f"Activations should be 2D, but got {activations.shape}"
            return activations
        batch = self.compute_batch()
        batches = []
        for i in tqdm(range(len(self.dataloader)), total=len(self.dataloader), desc="Computing activations..."):
            if batch == None: break
            batches.append(batch[1])
            batch = self.compute_batch()
        self.activations = t.concat(batches)
        if self.save_activation_path:
            print(f"Saving activations to {self.save_activation_path}")
            t.save(self.activations, self.save_activation_path)
        self.activations_computed = True
        return self.activations


    def compute_batch(self):
        # Main function to be overridden by child classes
        raise NotImplementedError("Subclasses should implement this method")
 

    @property
    def config(self):
        return {
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }
    
class OpenAICLIPActivationBuffer(CLIPActivationBuffer):
    def compute_batch(self):
        self.model.eval()
        with t.no_grad():
            try:
                batch = next(self.iterator)
                if self.modality == 'text':
                    inputs = clip.tokenize(batch).to(self.device)
                    outputs = self.model.encode_text(inputs)
                else:
                    inputs = [self.processor(x) for x in batch]
                    #inputs = vmap(self.processor)(batch).to(self.device)
                    inputs = t.tensor(np.stack(inputs)).to(self.device)
                    outputs = self.model.encode_image(inputs)
                if self.layer is not None:
                    outputs = self.latent_representations['act']
                    
                if self.activations == None:
                    self.activations = outputs
                else: 
                    self.activations = t.concat((self.activations, outputs), dim=0)
                return batch, outputs
            except StopIteration:
                return None
            




class MedCLIPActivationBuffer(CLIPActivationBuffer):
    def compute_batch(self):
        self.model.eval()
        with t.no_grad():
            try :
                batch = next(self.iterator)
                if self.modality == 'text':
                    inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    outputs = self.model.encode_text(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                else:
                    inputs = self.processor(images=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    outputs = self.model.encode_image(pixel_values=inputs['pixel_values'])
                if self.layer is not None:
                    outputs = self.latent_representations['act']
                if self.activations == None:
                    self.activations = outputs
                else: 
                    self.activations = t.concat((self.activations, outputs), dim=0)
                return batch, outputs
            except StopIteration:
                return None



class ActivationDataset(t.utils.data.Dataset):
    def __init__(self, activations):
        self.activations = activations.squeeze(0)

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]

class FeatureBuffer:
    """
    Implements a buffer of features of a SAE. The buffer stores activations from a model,
    yields them in batches, and refreshes them when full.
    """
    def __init__(self, 
                 activation_buffer, # Activation Buffer with the same batch size than out_batch_size
                 sae, # Dictionnary
                 out_batch_size=8192, # size of batches in which to yield activations
                 device='cpu', # device on which to store the activations
                 precompute=False,
                 load_features_path=None, # If you want to load already computed features
                 save_features_path=None, # If you want to save the computed features (None : you don't save, str : saving path)
                 return_raw_data=False # If you want to access the raw data corresponding to the features (e.g. FeatureBuffer)
                 ):
        
        self.activation_buffer = activation_buffer
        self.sae = sae
        if len(activation_buffer) == 0:
            raise Exception("Activation_buffer is empty !")
        self.sae.to(device)
        self.out_batch_size = out_batch_size
        self.device = device
        self.load_features_path = load_features_path
        self.save_features_path = save_features_path
        self.return_raw_data = return_raw_data
        self.features_computed = False
        self.precompute_idx = 0
        self.features = None
        if precompute or load_features_path:
            self.features = self.precompute_features()
            
        
    
    def __len__(self):
        if self.features_computed:
            return math.ceil(len(self.features)/self.out_batch_size)
        else: return len(self.activation_buffer)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of features
        """
        if not(self.features_computed):    
            with t.no_grad():
                outputs = self.compute_batch()
                if outputs == None:
                    self.refresh()
                    raise StopIteration
                return outputs
        else:
            if self.__len__() > self.precompute_idx:
                features = self.features[self.precompute_idx*self.out_batch_size:(self.precompute_idx+1)*self.out_batch_size]
                self.precompute_idx += 1
                if self.return_raw_data:
                    return self.activation_buffer.retrieve_batch_raw(), features
                else: return features
            else:
                self.refresh()
                raise StopIteration
    
    def refresh(self):
        self.precompute_idx = 0
        self.features_computed = True
        if self.save_features_path:
            print(f"Saving features to {self.save_features_path} ...")
            t.save(self.features.detach().cpu(), self.save_features_path)
            self.save_features_path = None
    
    
    def precompute_features(self):
        if self.load_features_path:
            print(f"Loading features from {self.load_features_path} ...")
            self.features_computed = True
            out = t.load(self.load_features_path)
            assert len(out.shape) == 2, f"Features should be 2D, but got {out.shape}"
            return out.to(self.device)
        batch = self.compute_batch()
        batches = []
        while batch != None:
            batches.append(batch[1]) if self.return_raw_data else batches.append(batch)
            batch = self.compute_batch()
        self.features = t.concat(batches)
        if self.save_features_path:
            print(f"Saving features to {self.save_features_path} ...")
            t.save(self.features.detach().cpu(), self.save_features_path)
        self.features_computed = True
        return self.features


    def compute_batch(self):
        self.sae.eval()
        with t.no_grad():
            try:
                batch = next(self.activation_buffer)
                if self.return_raw_data:
                    if not(isinstance(batch,tuple)):
                        print("WARNING : Be careful, the activation buffer does not provie the raw data !")
                        batch = (t.zeros((batch.shape[0], 1)), batch.to(self.device))
                    else:
                        to_tensor = torchvision.transforms.ToTensor()
                        imgs = [to_tensor(x).unsqueeze(0).to(self.device) for x in batch[0]]
                        imgs = t.cat(imgs, dim=0).to(self.device)
                        batch = (imgs, batch[1].to(self.device))
                    self.sae.to(dtype=batch[1].dtype)
                    outputs = self.sae.encode(batch[1])
                else:
                    batch.to(self.device)
                    self.sae.to(dtype=batch.dtype)
                    outputs = self.sae.encode(batch)
                
                if self.features == None:
                    self.features = outputs
                else: 
                    self.features = t.concat((self.features, outputs), dim=0)
                self.precompute_idx += 1
                if self.return_raw_data:
                    return batch[0], outputs
                return outputs
            except StopIteration:
                return None

   # def get_batch(self, modality='img'):
    #     batch = []
    #     count = 0
    #     while count < self.out_batch_size:
    #         if modality=='img':
    #             if self.img_idx < len(self.img_data):
    #                 batch.append(self.img_data[self.img_idx]) 
    #                 self.img_idx += 1
    #             else: break
    #         if modality=='text':
    #             if self.txt_idx < len(self.txt_data):
    #                 batch.append(self.txt_data[self.txt_idx])
    #                 self.txt_idx += 1
    #             else: break
    #         count += 1
    #     size = len(batch)
    #     if size != 0:
    #         for i in range(self.out_batch_size-size):
    #             batch.append(batch[i%size])
    #     return batch

    # def __next__(self):
    #     """
    #     Return a batch of activations
    #     """
    #     if not(self.activations_computed):    
    #         self.model.eval()
    #         with t.no_grad():
    #             batch = self.get_batch()
    #             if len(batch) == 0:
    #                 batch = self.get_batch(modality='text')
    #                 if len(batch) == 0:
    #                     self.refresh()
    #                     return self.__next__()
    #                 inputs = self.processor(texts=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
    #                 outputs = self.model.encode_text(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    #             else:
    #                 inputs = self.processor(images=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
    #                 outputs = self.model.encode_image(pixel_values=inputs['pixel_values'])
    #             if self.activations == None:
    #                 self.activations = outputs.unsqueeze(0)
    #             else: 
    #                 self.activations = t.concat((self.activations, outputs.unsqueeze(0)), dim=0)
    #             return outputs
    #     else:
    #         if len(self.activations) > self.precompute_idx:
    #             outputs = self.activations[self.precompute_idx]
    #             self.precompute_idx += 1
    #             return outputs
    #         else:
    #             self.refresh()
    #             return self.__next__()

    # def to_torch_dataloader(self, batch_size=0):
    #     dataset = ActivationDataset(self.activations)
    #     if batch_size == 0: batch_size = self.out_batch_size
    #     dataloader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #     return dataloader
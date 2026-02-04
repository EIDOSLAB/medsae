


from torch.utils.data import Dataset
import torch as t
import torchvision
import data_preprocess
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer
import clip
from tqdm.auto import tqdm
import argparse
import config
import os
import utils
import pandas as pd
import json
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

from huggingface_hub import get_token
from transformers import pipeline
from huggingface_hub import notebook_login
from transformers import BitsAndBytesConfig
import ast
import numpy as np








SYSTEM_INSTR = """Act as a precise and analytical radiologist.
"""


def stratified_sampling_by_deciles_(features_dataset, n_activating=100, n_non_activating=100, seed=42):
    """
    For each latent (dimension), sample:
    - n_activating examples stratified across deciles (evenly)
    - n_non_activating examples (from lower 50% by default)

    Args:
        features_dataset (Tensor): [n_samples, dim] tensor
        n_activating (int): total number of activating samples (must be divisible by 10)
        n_non_activating (int): total number of non-activating samples
        seed (int): random seed

    Returns:
        activating_samples: dict[dim] = indices of activating samples
        non_activating_samples: dict[dim] = indices of non-activating samples
    """
    assert n_activating % 10 == 0, "n_activating must be divisible by 10 (for deciles)."
    
    t.manual_seed(seed)
    np.random.seed(seed)

    n_samples, dim = features_dataset.shape
    features_np = features_dataset.cpu().numpy()

    activating_samples = {}
    non_activating_samples = {}

    samples_per_decile = n_activating // 10

    for i in range(dim):
        activations = features_np[:, i]
        # Compute decile edges (10 bins)
        decile_edges = np.percentile(activations, np.arange(0, 101, 10))

        stratified_idxs = []
        for j in range(10):
            if j < 9:
                mask = (activations >= decile_edges[j]) & (activations < decile_edges[j + 1])
            else:
                mask = (activations >= decile_edges[j]) & (activations <= decile_edges[j + 1])
            decile_indices = np.where(mask)[0]

            if len(decile_indices) >= samples_per_decile:
                sampled = np.random.choice(decile_indices, size=samples_per_decile, replace=False)
            else:
                sampled = np.random.choice(decile_indices, size=samples_per_decile, replace=True)

            stratified_idxs.extend(sampled)

        activating_samples[i] = np.array(stratified_idxs)

        # Non-activating samples from the lowest 50%
        threshold = np.percentile(activations, 50)
        low_activation_indices = np.where(activations <= threshold)[0]

        if len(low_activation_indices) >= n_non_activating:
            sampled_low = np.random.choice(low_activation_indices, size=n_non_activating, replace=False)
        else:
            sampled_low = np.random.choice(low_activation_indices, size=n_non_activating, replace=True)

        non_activating_samples[i] = sampled_low

    return activating_samples, non_activating_samples

def stratified_sampling_by_deciles(features_dataset, n_activating=100, n_non_activating=100, seed=42):
    """
    For each latent (dimension), sample:
    - n_activating examples stratified across deciles (evenly)
    - n_non_activating examples (from lower 50% by default)

    Args:
        features_dataset (Tensor): [n_samples, dim] tensor
        n_activating (int): total number of activating samples (must be divisible by 10)
        n_non_activating (int): total number of non-activating samples
        seed (int): random seed

    Returns:
        activating_samples: dict[dim] = indices of activating samples
        non_activating_samples: dict[dim] = indices of non-activating samples
    """
    assert n_activating % 10 == 0, "n_activating must be divisible by 10 (for deciles)."
    
    t.manual_seed(seed)
    np.random.seed(seed)

    n_samples, dim = features_dataset.shape
    features_np = features_dataset.cpu().numpy()

    activating_samples = {}
    non_activating_samples = {}

    base_samples_per_decile = n_activating // 10

    for i in range(dim):
        activations = features_np[:, i]
        decile_edges = np.percentile(activations, np.arange(50, 101, 5))

        # Collect non-empty deciles
        decile_indices_list = []
        for j in range(5):
            if j < 4:
                mask = (activations >= decile_edges[j]) & (activations < decile_edges[j + 1])
            else:
                mask = (activations >= decile_edges[j]) & (activations <= decile_edges[j + 1])
            indices = np.where(mask)[0]
            if len(indices) > 0:
                decile_indices_list.append(indices)

        n_valid_deciles = len(decile_indices_list)
        if n_valid_deciles == 0:
            activating_samples[i] = np.array([], dtype=int)
            non_activating_samples[i] = np.array([], dtype=int)
            continue

        samples_per_decile = n_activating // n_valid_deciles
        stratified_idxs = []

        for indices in decile_indices_list:
            if len(indices) >= samples_per_decile:
                sampled = np.random.choice(indices, size=samples_per_decile, replace=False)
            else:
                print(f"Warning: Decile {j} for feature {i} has only {len(indices)} samples, sampling with replacement.")
                sampled = np.random.choice(indices, size=samples_per_decile, replace=True)
            stratified_idxs.extend(sampled)

        # If rounding led to fewer than requested samples, top up
        if len(stratified_idxs) < n_activating:
            remaining = n_activating - len(stratified_idxs)
            flat_all = np.concatenate(decile_indices_list)
            extra = np.random.choice(flat_all, size=remaining, replace=(len(flat_all) < remaining))
            stratified_idxs.extend(extra)

        activating_samples[i] = np.array(stratified_idxs)

        # Non-activating: 
        threshold = 1e-7  # Set a threshold for low activation
        low_activation_indices = np.where(activations <= threshold)[0]

        if len(low_activation_indices) >= n_non_activating:
            sampled_low = np.random.choice(low_activation_indices, size=n_non_activating, replace=False)
        else:
            sampled_low = np.random.choice(low_activation_indices, size=n_non_activating, replace=True)

        non_activating_samples[i] = sampled_low

    return activating_samples, non_activating_samples


def sampling(features_dataset, n_activating=100, n_non_activating=100, seed=42):
    """
    For each latent (dimension), sample:
    - n_activating examples stratified across deciles (evenly)
    - n_non_activating examples (from lower 50% by default)

    Args:
        features_dataset (Tensor): [n_samples, dim] tensor
        n_activating (int): total number of activating samples (must be divisible by 10)
        n_non_activating (int): total number of non-activating samples
        seed (int): random seed

    Returns:
        activating_samples: dict[dim] = indices of activating samples
        non_activating_samples: dict[dim] = indices of non-activating samples
    """
    
    t.manual_seed(seed)
    np.random.seed(seed)

    n_samples, dim = features_dataset.shape
    features_np = features_dataset.cpu().numpy()

    activating_samples = {}
    non_activating_samples = {}

    

    for i in range(dim):
        activations = features_np[:, i]

        threshold = 1e-7  # Set a threshold for low activation
        activation_indices = np.where(activations > threshold)[0]
        high_activation_indices = np.argsort(activations)[::-1]  # Sort indices by activation value in descending order
        high_activation_indices = high_activation_indices[:min(len(high_activation_indices), n_activating)]

        if len(high_activation_indices) == 0:
            print(f"Warning: feature {i} has no activating samples, skipping.")
            activating_samples[i] = activating_samples[i-1]
            non_activating_samples[i] = non_activating_samples[i-1] 
            continue

        if len(high_activation_indices) >= n_activating:
            sampled = np.random.choice(high_activation_indices, size=n_activating, replace=False)
            #sampled += np.random.choice(activation_indices, size=n_activating//2, replace=False)
        else:
            print(f"Warning: feature {i} has only {len(high_activation_indices)}  activating samples, sampling with replacement.")
            sampled = np.random.choice(high_activation_indices, size=n_activating, replace=True)
            #sampled += np.random.choice(activation_indices, size=n_activating//2, replace=True)
        
        activating_samples[i] = sampled

        # Non-activating: 
        threshold = 1e-7  # Set a threshold for low activation
        low_activation_indices = np.where(activations <= threshold)[0]

        if len(low_activation_indices) == 0:
            print(f"Warning: feature {i} has no non-activating samples, skipping.")
            non_activating_samples[i] = non_activating_samples[i-1] 
            continue

        if len(low_activation_indices) >= n_non_activating:
            sampled_low = np.random.choice(low_activation_indices, size=n_non_activating, replace=False)
        else:
            print(f"Warning: feature {i} has only {len(low_activation_indices)}  non-activating samples, sampling with replacement.")
            sampled_low = np.random.choice(low_activation_indices, size=n_non_activating, replace=True)

        non_activating_samples[i] = sampled_low

    return activating_samples, non_activating_samples

def init_medgemma():
    if get_token() is None:
        notebook_login()

    model_variant = "4b-it"  # @param ["4b-it", "27b-text-it"]
    model_id = f"google/medgemma-{model_variant}"

    use_quantization = True  # @param {type: "boolean"}

    # @markdown Set `is_thinking` to `True` to turn on thinking mode. **Note:** Thinking is supported for the 27B variant only.
    is_thinking = False  # @param {type: "boolean"}

    # If running the 27B variant in Google Colab, check if the runtime satisfies
    # memory requirements
    if "27b" in model_variant and google_colab:
        if not ("A100" in t.cuda.get_device_name(0) and use_quantization):
            raise ValueError(
                "Runtime has insufficient memory to run the 27B variant. "
                "Please select an A100 GPU and use 4-bit quantization."
            )

    model_kwargs = dict(
        torch_dtype=t.bfloat16,
        device_map="auto",
    )

    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)


    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        model_kwargs=model_kwargs,
    )

    pipe.model.generation_config.do_sample = False

    return pipe


def get_activating_guess(pipe, images, concept, description):
    messages = []
    bs = 5
    PROMPT = f"""
    You will receive a specific concept and a description, such as "Pulmonary edema and pleural effusions in a patient with a central line" or "Technical Artifacts and/or Equipment-Related Anomalies."
    Following this, you will be presented with multiple image examples.
    Your objective is to assess which of these examples accurately contains the given concept.
    For each image example provided, indicate with a 1 if the image is correctly labeled, or a 0 if it is mislabeled.
    Ensure your response is formatted strictly as a valid Python list of length {bs}.
    Return only the Python list without any additional information.

    Concept: {concept}
    Description: {description}
    Image examples:
    """
    for i in range(0, len(images), bs):
        
        content = [{"type": "text", "text": PROMPT}]
        # Iterate over the DataFrame and append image and text entries
        image_entry = [{"type": "image", "image": img} for img in images[i:min(i+bs, len(images))]]
        content += image_entry


        # Construct the messages list
        messages.append([
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_INSTR}]
            },
            {
                "role": "user",
                "content": content
            }
        ])
    
    

    activations = []
        
    for i, out in enumerate(pipe(text=messages, max_new_tokens=500, batch_size=10)):
        output = out[0]["generated_text"][-1]["content"]
        output = output.strip('```python\n').strip('\n```')
        try:
            out = ast.literal_eval(output)
            if len(out) != len(messages[i][1]['content']) - 1:  # -1 for the prompt
                raise ValueError(f"Output length {len(out)} does not match required size {len(messages[i][1]['content']) - 1}.")
        except ValueError as ve:
            print(f"ValueError: {ve}")
            out = [0] * bs
        except:
            print("ERROR STRING: ", output)
            out = [0] * bs
        activations = activations + out
        
    return activations


def main(args):
    sae_path = os.path.join(config.SAE_CKPTS,args.sae_path)

    (SaeArchitecture, _) = utils.get_sae_architecture(args.sae_type_name)
    device = args.device 

    sae = SaeArchitecture.from_pretrained(sae_path).to(device)
    activation_dim = sae.activation_dim
    dict_size = sae.dict_size
    expansion_factor = dict_size/activation_dim
    

    batch_size = 1024
    model_name = args.model_name
    device=args.device
    #dataloader = utils.get_dataset(args.dataset, batch_size)
    dataloader = utils.get_dataset(args.dataset, batch_size, labels=False, shuffled_version=args.shuffle_dataset)
    #Model import
    print(f"Importing model {model_name}...")
    #model, processor = utils.get_model(model_name)
    model, processor = t.Tensor([]), None
    #Buffer set-up
    print(f"Setting up activation buffer...")
    (precomputed, path) = utils.activations_already_precomputed(model_name, args.dataset, args.split, args.layer)

    activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader[args.split],
        modality='img',
        model=model,
        layer=None if not(args.layer) else utils.rgetattr(model.vision_model, args.layer),
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        return_raw_data=False,
        load_activations_path=args.load_activations_path if args.load_activations_path else (path if precomputed else None),
        save_activations_path=args.save_activations_path if args.save_activations_path else (path if not(precomputed) else None),
    )
    sae_name = args.sae_path.split('.pt')[0]
    (precomputed, path) = utils.features_already_precomputed(sae_name, args.dataset, args.split, args.layer)
    print(f"Setting up feature buffer...")
    feature_buffer = FeatureBuffer(
        activation_buffer=activation_buffer,
        sae=sae,
        out_batch_size=batch_size,
        device=device,
        load_features_path=args.load_features_path if args.load_features_path else (path if precomputed else None),
            save_features_path=args.save_features_path if args.save_features_path else (path if not(precomputed) else None),
        return_raw_data=False,
    )

    total = len(feature_buffer)
    features_dataset = None

    for i, features in tqdm(enumerate(feature_buffer), total=total, desc="Computing SAE Features"):
        if features_dataset is None:
            features_dataset = features.to(device)
        else:
            features_dataset = t.cat((features_dataset, features.to(device)), dim=0)

    #features_dataset = activation_buffer.activations
    activating_samples, non_activating_samples = sampling(
        features_dataset, 
        n_activating=args.img_per_latent, 
        n_non_activating=args.img_per_latent,
    )
    

    features_dataset[features_dataset < 1e-7] = 0
    features_dataset[features_dataset >= 1e-7] = 1
    l0 = t.sum((features_dataset.abs() > 1e-7).to(t.float32), dim=0)
    
    variances = t.var(features_dataset, dim=0)
    # Identify dead features with zero variance
    alive_features_indices = t.where(variances != 0)[0]
    alive_features_indices = [
        1215, 7468, 2320, 876, 85, 3826, 4214, 283, 817, 1859,
        5929, 127, 5616, 1251, 2378, 2862, 8153, 4728, 1266, 2708, 6552
    ]

    df = pd.read_csv(os.path.join(config.CONCEPTS, args.assigned_concepts))
    neurons = df['neuron'].tolist()
    concepts = df['concepts'].tolist()
    descriptions = df['description'].tolist()

    precision = []
    recall = []
    f1 = []

    # Initialize MedGemma
    pipe = init_medgemma()

    output_dir = os.path.join(config.RESULTS, args.sae_path.split('.')[0])
    # Save the results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{args.assigned_concepts}_eval_results.csv")

    with open(output_path, 'w') as f:
        f.write("neuron,precision,recall,f1,concept,description\n")
        for i, neuron in tqdm(enumerate(neurons), desc="Evaluating features"):
            if neuron not in alive_features_indices:
                #print(f"Feature {neuron} is dead (zero variance). Skipping.")
                precision.append(0.0)
                recall.append(0.0)
                f1.append(0.0)
                continue

            activating_indices = activating_samples[neuron]
            non_activating_indices = non_activating_samples[neuron]

            activating_list = activating_indices.tolist() if isinstance(activating_indices, np.ndarray) else activating_indices
            non_activating_list = non_activating_indices.tolist() if isinstance(non_activating_indices, np.ndarray) else non_activating_indices
        

            images = [dataloader[args.split].dataset[j] for j in activating_list + non_activating_list]        
            target = [features_dataset[j, neuron].cpu().item() for j in activating_list + non_activating_list]
            
            # shuffle the images and target together
            indices = np.arange(len(images))
            np.random.shuffle(indices)
            images = [images[idx] for idx in indices]
            target = [target[idx] for idx in indices]

            

            


            concept = concepts[i]
            description = descriptions[i]
            # Get the activations from MedGemma
            activations = get_activating_guess(pipe, images, concept, description)
            print(f"Feature {neuron} LO : {l0[neuron]} - Concept : {concept}\n Activations ({np.sum(activations)}): {activations} \n Target ({np.sum(target)}): {target}")

            precision.append(precision_score(target, activations))
            recall.append(recall_score(target, activations))
            f1.append(f1_score(target, activations))
            f.write(f"{neuron},{precision[-1]},{recall[-1]},{f1[-1]},\"{concept}\",\"{description}\"\n")
            print(f"Precision: {precision[-1]:.4f}, Recall: {recall[-1]:.4f}, F1: {f1[-1]:.4f}")


    results = {
        "neuron": neurons,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "concepts": concepts,
        "descriptions": descriptions,
    }

    print(f"Precision: {np.mean(precision):.4f}, Recall: {np.mean(recall):.4f}, F1: {np.mean(f1):.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=True)
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that plots the top activating images corresponding to the SAE's features.")
    parser.add_argument('--sae_path', type=str, required=False, default="ae.pt", help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="val")
    parser.add_argument('--img_per_latent', type=int, required=False, help=f'The number of activating and non activating images used for evaluation per latent.', default=30)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_features_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the sae's features starting from {config.FEATURES}.", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features starting from {config.FEATURES}.", default=None)
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)
    parser.add_argument('--shuffle_dataset', type=bool, required=False, help=f'To use shuffle dataset version', default=False)
    parser.add_argument('--assigned_concepts', type=str, required=False, help=f'The path of the .csv file containing the assigned concepts for each latent. If not provided, it will use the default concepts.')


    args = parser.parse_args()
    main(args)





from torch.utils.data import Dataset
import torch as t
import data_preprocess
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer
import clip
from tqdm.auto import tqdm
import argparse
import config
import utils
import os
import pandas as pd



def main(args):
    sae_path = os.path.join(config.SAE_CKPTS,args.sae_path)
    sae_name =args.sae_path.split(".")[0]
    model_name = args.model_name
    batch_size = args.batch_size
    device=args.device

    (SaeArchitecture, _) = utils.get_sae_architecture(args.sae_type_name)

    sae = SaeArchitecture.from_pretrained(sae_path).to(device)
    activation_dim = sae.activation_dim
    dict_size = sae.dict_size
    expansion_factor = dict_size/activation_dim



    vocab_path = os.path.join(config.VOCABS,args.vocab_path)
    vocab_name = vocab_path.split("/")[-1].split(".")[0]
    dataloader = data_preprocess.import_vocab(vocab_path, batch_size)


    #Model import
    print(f"Importing model {model_name}...")
    model, processor = utils.get_model(model_name)

    #Buffer set-up
    print(f"Setting up activation buffer...")

    activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader,
        modality='text',
        model=model,
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        return_raw_data=True,
        load_activations_path=args.load_activations_path if args.load_activations_path else None,
        save_activations_path=args.save_activations_path if args.save_activations_path else None,
    )

    total = len(activation_buffer)
    names = []
    vocab_embeddings = []
    for i, (name, vocab_embedding) in tqdm(enumerate(activation_buffer), total=total, desc="Embedding vocabulary"):
        if i == total: break
        names += name
        vocab_embeddings.append(vocab_embedding)
    vocab_embeddings = t.vstack(vocab_embeddings).to(t.float32)
    vocab_embeddings = vocab_embeddings - vocab_embeddings.mean(dim=0, keepdim=True)

    decoder = sae.decoder.weight.detach().squeeze().to(t.float32) + sae.decoder.bias.detach().squeeze().to(t.float32).unsqueeze(1) # (n_features, n_latents)
    print(vocab_embeddings.shape)
    decoder = decoder - decoder.mean(dim=1, keepdim=True)
    decoder_norm = decoder/decoder.norm(dim=0)[None, :]
    vocab_embeddings_norm = vocab_embeddings/vocab_embeddings.norm(dim=1)[:, None]
    similarity_matrix = t.mm(vocab_embeddings_norm, decoder_norm).detach().cpu()
    idxs = similarity_matrix.argmax(axis=0)
    top_concept_names = pd.DataFrame({'concept_name':[names[i] for i in idxs], 'sim_value':[similarity_matrix[idxs[i]][i].item() for i in range(similarity_matrix.shape[1])]})
    output_path = os.path.join(config.CONCEPTS,f"{sae_name}_{vocab_name}.csv")
    print(f"Saving to : {output_path}")
    top_concept_names.to_csv(output_path, index_label="idx")
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that assign names to the features of the SAE (the result is saved in {config.CONCEPTS}).")
    parser.add_argument('--sae_path', type=str, required=False, default="ae.pt", help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--vocab_path', type=str, required=True, help=f'The path of the .txt vocabulary file.')
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the vocab activations', default=1024)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    


    args = parser.parse_args()
    main(args)
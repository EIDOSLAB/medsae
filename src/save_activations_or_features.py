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
    model_name = args.model_name
    batch_size = args.batch_size
    device=args.device

    if args.vocab_path:
        vocab_path = args.vocab_path
        dataset_name = vocab_path.split("/")[-1].split(".")[0]
        dataloader = data_preprocess.import_vocab(vocab_path, batch_size)
    else:   
        dataset_name = args.dataset
        dataloader = utils.get_dataset(args.dataset, batch_size)[args.split]

    #Model import
    print(f"Importing model {model_name}...")
    model, processor = utils.get_model(model_name)

    #Buffer set-up
    print(f"Setting up activation buffer...")
    (precomputed, path) = utils.activations_already_precomputed(model_name, dataset_name, args.split, args.layer)

    activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader,
        modality='text' if args.vocab_path else 'img',
        model=model,
        layer=None if args.vocab_path or not(args.layer) else utils.rgetattr(model.vision_model, args.layer),
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        load_activations_path=args.load_activations_path if args.load_activations_path else (path if precomputed else None),
        save_activations_path=args.save_activations_path if args.save_activations_path else (path if not(precomputed) else None),
    )

    if args.sae_path is not None:
        sae_path = os.path.join(config.SAE_CKPTS,args.sae_path)
        (SaeArchitecture, _) = utils.get_sae_architecture(args.sae_type_name)

        sae = SaeArchitecture.from_pretrained(sae_path).to(device)
        activation_dim = sae.activation_dim
        dict_size = sae.dict_size
        expansion_factor = dict_size/activation_dim
        sae_name = args.sae_path.split('.pt')[0]

        print(f"Setting up feature buffer...")

        (precomputed, path) = utils.features_already_precomputed(sae_name, dataset_name, args.split, args.layer)
        feature_buffer = FeatureBuffer(
            activation_buffer=activation_buffer,
            sae=sae,
            out_batch_size=batch_size,
            device=device,
            load_features_path=None,
            save_features_path=args.save_features_path if args.save_features_path else (path if not(precomputed) else None),
        )

        for i, _ in tqdm(enumerate(feature_buffer), total=len(feature_buffer), desc="Computing SAE Features" if activation_buffer.load_activations else "Computing SAE Features and dataset activations"):
            pass
    else:   
        for i, _ in tqdm(enumerate(activation_buffer), total=len(activation_buffer), desc="Computing dataset activations"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that computes activations or/and corresponding SAE's features.")
    parser.add_argument('--sae_path', type=str, required=False, default=None, help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the vocab activations', default=1024)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features. If not provided it will automatically save the sae's features to {config.FEATURES}.", default=None)
    parser.add_argument('--vocab_path', type=str, required=False, help=f'The path of the .txt vocabulary file if you want to compute vocab activations/features.', default=None)
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="")   
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)

    args = parser.parse_args()
    main(args)
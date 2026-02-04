from torch import save
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



def main():
    model_name = "MedCLIP-RN50"
    batch_size = 512
    device='cuda'
    dataset_name = "chexpert"
    split = 'trainval'
    layer = None
    dataloader = utils.get_dataset(dataset_name, batch_size)[split]

    #Model import
    print(f"Importing model {model_name}...")
    model, processor = utils.get_model(model_name)

    #Buffer set-up
    print(f"Setting up activation buffer...")
    (precomputed, path) = utils.activations_already_precomputed(model_name, dataset_name, split, layer)

    activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader,
        modality='img',
        model=model,
        layer=None if not(layer) else utils.rgetattr(model.vision_model, layer),
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        load_activations_path=(path if precomputed else None),
        save_activations_path=(path if not(precomputed) else None),
    )

    folder_path = os.path.join(config.SAE_CKPTS,"sweep",model_name,dataset_name)
    nb_files = len(os.listdir(folder_path))
    for i,file in enumerate(os.listdir(folder_path)):
        print(f"SAE {i+1}/{nb_files}..")
        print(file)
        sae_path = os.path.join(folder_path,file,"ae.pt")
        (SaeArchitecture, _) = utils.get_sae_architecture(config.SAE_TRAINER_TO_SAE[file.split('-')[0]])

        sae = SaeArchitecture.from_pretrained(sae_path).to(device)
        activation_dim = sae.activation_dim
        dict_size = sae.dict_size
        expansion_factor = dict_size/activation_dim

        print(f"Setting up feature buffer...")

        output = os.path.join(config.FEATURES,"sweep",file+'.pt')
        if os.path.exists(output)
            load_features_path = output
            save_features_path = None
        else: 
            save_features_path = output
            load_features_path = None

        feature_buffer = FeatureBuffer(
            activation_buffer=activation_buffer,
            sae=sae,
            out_batch_size=batch_size,
            device=device,
            load_features_path=load_features_path,
            save_features_path=load_features_path,
        )

        for i, _ in tqdm(enumerate(feature_buffer), total=len(feature_buffer), desc=f"Computing SAE Features of {file}" if activation_buffer.load_activations else "Computing SAE Features and dataset activations"):
            pass

if __name__ == "__main__":
    main()
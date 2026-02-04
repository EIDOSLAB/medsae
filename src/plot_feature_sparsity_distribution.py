import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import data_preprocess
import argparse
import utils
import clip
import torch as t
from dictionary_learning.buffer import CLIPActivationBuffer
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
import config
import os
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors

from dictionary_learning.dictionary import AutoEncoder, AutoEncoderNew
from tqdm import tqdm
import pandas as pd


k = 100

def feature_density_diagram(features, concept_sim_values=None):
    """
    Compute the table of the log10 latent sparsity (how often it fires) for all latents.
    """
    # Compute the density of each feature
    log_density = t.log10(t.mean((features != 0).to(t.float32), dim=0) + 1e-10).cpu().numpy()
    kde = stats.gaussian_kde(log_density)
    x_vals = np.linspace(min(log_density), max(log_density), 100)  # Range for x axis
    y_vals = kde(x_vals)  # Compute KDE for each x value
    if concept_sim_values is not None:
        # Interpolate the concept similarity values to match the x_vals
        interp_func = interp1d(log_density, concept_sim_values, kind='nearest', bounds_error=False, fill_value="extrapolate")
        colors = interp_func(x_vals)
        colors = colors ** 3
        colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
        topksim_log_density = log_density[np.argsort(concept_sim_values)[-k:]]
        kde_topk = stats.gaussian_kde(topksim_log_density)
        y_vals_topk = kde_topk(x_vals)
        return x_vals, y_vals, colors, y_vals_topk
    return x_vals, y_vals


def main(args):
    model_name = args.model_name
    batch_size = args.batch_size
    device = args.device

    #Dataset preprocess
    dataloader = utils.get_dataset(args.dataset, batch_size, labels=False)[args.split]

    #Model import
    print(f"Importing model {model_name}...")
    model, processor = clip.load("RN50")

    print(f"Setting up activation buffer...")
    (precomputed, path) = utils.activations_already_precomputed(model_name, args.dataset, args.split, args.layer)

    val_activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader,
        modality='img',
        model=model,
        layer=None if not(args.layer) else utils.rgetattr(model.vision_model, args.layer),
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        load_activations_path=args.load_activations_path if args.load_activations_path else (path if precomputed else None),
        shuffle=False,
    )
    if args.sae_path:
        sae_name = args.sae_path
        sae_path = config.SAE_CKPTS+f"/{sae_name}"

        (SAEArchitecture, _) = utils.get_sae_architecture(args.sae_type_name)
        sae = SAEArchitecture.from_pretrained(sae_path).to(device)

        features = t.empty((0, sae.dict_size), dtype=t.float32, device='cuda')

        with t.no_grad():
            for x in tqdm(val_activation_buffer, total=len(val_activation_buffer)):
                x = x.to(device).to(dtype=t.float32)
                _, f = sae(x, output_features=True)
                features_BF = t.flatten(f, start_dim=0, end_dim=-2).to(dtype=t.float32)
                features = t.cat((features,features_BF), dim=0)
        os.makedirs(config.RESULTS+f"/{sae_name.split('.')[0]}", exist_ok=True)
        path = os.path.join(config.RESULTS, f"{sae_name.split('.')[0]}", f"feature_sparsity_distribution_{args.dataset}_{args.split}.png")
    else:
        for _ in tqdm(val_activation_buffer, total=len(val_activation_buffer)):
            pass
        features = val_activation_buffer.activations
        os.makedirs(config.RESULTS+f"/{model_name.lower()}", exist_ok=True)
        path = os.path.join(config.RESULTS, f"{model_name.lower()}", f"feature_sparsity_distribution_{args.dataset}_{args.split}.png")

    if args.assigned_concepts is not None:
        concept_sim_values = pd.read_csv(config.CONCEPTS+f"/{args.assigned_concepts}")
        concept_sim_values = concept_sim_values['sim_value'].to_numpy()
        x,y,colors,y_topk = feature_density_diagram(features, concept_sim_values)

        #Save a curve in png :
        plt.scatter(x, y, c=colors, cmap='viridis', s=10)
        plt.colorbar(label='Concept Similarity')
        plt.title("Feature Sparsity Distribution")
        plt.xlabel("Log10 Latent Sparsity")
        plt.ylabel("Proportion of Latents")

        # Add a curve for the top-k features:
        plt.plot(x, y_topk, color='red', label=f"Top{k} Features in naming similarity score")
        plt.legend()
        
    else:
        x,y = feature_density_diagram(features)
        #Save a curve in png :
        plt.plot(x, y)
        plt.title("Feature Sparsity Distribution")
        plt.xlabel("Log10 Latent Sparsity")
        plt.ylabel("Proportion of Latents")
    
    plt.savefig(path)
    print(f"Saving to : {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that plots the Feature Sparsity Distribution of a given SAE.")
    parser.add_argument('--sae_path', type=str, required=False, default=None, help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the vocab activations', default=1024)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features. If not provided it will automatically save the sae's features to {config.FEATURES}.", default=None)
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="val") 
    parser.add_argument('--assigned_concepts', type=str, required=False, help=f'[Optional] The path of the .csv file of the concept names/features association table starting from {config.CONCEPTS}. It allows to add a color to the diagram showing similarities between the features and the concepts.', default=None)   
    parser.add_argument('--layer', type=str, required=False, help=f'[Optional] The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)

    args = parser.parse_args()
    main(args)
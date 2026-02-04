import torch
import os.path
import argparse
import os
import pandas as pd
import config
import utils
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer

METRICS = ['correlation', 'precision', 'recall', 'F1']

def get_args_parser():
    parser = argparse.ArgumentParser("Measure class correlation via weighted class appartenance.", add_help=False)
    parser.add_argument('--sae_path', type=str, required=False, default=None, help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the activations', default=512)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features. If not provided it will automatically save the sae's features to {config.FEATURES}.", default=None)
    parser.add_argument('--load_features_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the sae's features. If not provided it will automatically check if features can be loaded from {config.FEATURES}.", default=None)
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="trainval")   
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)
    parser.add_argument('--hierarchy_level', type=str, required=False, help=f"The hierarchy level of the labels to use. Either 'low' (bottom of the hierarchy), 'high' (top of the hierarchy) or 'both'. If not provided it will use the last level.", default='both')
    return parser

def main(args):
    model_name = args.model_name
    batch_size = args.batch_size
    device=args.device

    dataset_name = args.dataset
    dataloader, labels = utils.get_dataset(args.dataset, batch_size, labels=True, shuffled_version=True)
    dataloader = dataloader[args.split]
    labels = labels[args.split]

    print(f"Importing model {model_name}...")
    model, processor = utils.get_model(model_name)

    print(f"Setting up activation buffer...")
    (precomputed, path) = utils.activations_already_precomputed(model_name, dataset_name, args.split)

    

    activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader,
        modality='img',
        model=model,
        layer=None if args.layer is None else utils.rgetattr(model.vision_model, args.layer),
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        load_activations_path=args.load_activations_path if args.load_activations_path else (path if precomputed else None),
        save_activations_path=args.save_activations_path if args.save_activations_path else (path if not(precomputed) else None),
    )

    if args.sae_path is not None:
        sae_path = os.path.join(config.SAE_CKPTS,args.sae_path)
        sae_name = args.sae_path.split('.')[0]

        (SaeArchitecture, _) = utils.get_sae_architecture(args.sae_type_name)

        sae = SaeArchitecture.from_pretrained(sae_path).to(device)
        activation_dim = sae.activation_dim
        dict_size = sae.dict_size
        expansion_factor = dict_size/activation_dim

        print(f"Setting up feature buffer...")

        (precomputed, path) = utils.features_already_precomputed(sae_name, dataset_name, args.split, args.layer)
        feature_buffer = FeatureBuffer(
            activation_buffer=activation_buffer,
            sae=sae,
            out_batch_size=batch_size,
            device=device,
            load_features_path=args.load_features_path if args.load_features_path else (path if precomputed else None),
            save_features_path=args.save_features_path if args.save_features_path else (path if not(precomputed) else None),
        )


        for i, _ in tqdm(enumerate(feature_buffer), total=len(feature_buffer), desc=("Computing SAE Features" if activation_buffer.load_activations else "Computing SAE Features and dataset activations")):
                pass
        
        activations = feature_buffer.features
        output_dir = os.path.join(config.RESULTS, args.sae_path.split('.')[0])
    else:
        for i, _ in tqdm(enumerate(activation_buffer), total=len(activation_buffer), desc=("Computing activations")):
                pass
        activations = activation_buffer.activations.to(device)
        output_dir = os.path.join(config.RESULTS, model_name.lower())

    if args.split == 'trainval':
        classes = labels.dataset.dataset.labels[labels.dataset.indices].to(device)
        labels_name = labels.dataset.dataset.labels_name
    else:
        classes = labels.dataset.labels.to(device)
        labels_name = labels.dataset.labels_name

    
    if args.hierarchy_level == 'low':
        idx = [labels_name.index(label) for label in config.CHEXPERT_LOW_LABELS if label in labels_name]
        classes = classes[:, idx]
        labels_name = [labels_name[i] for i in idx]
    elif args.hierarchy_level == 'high':
        idx = [labels_name.index(label) for label in config.CHEXPERT_HIGH_LABELS if label in labels_name]
        classes = classes[:, idx]
        labels_name = [labels_name[i] for i in idx]
    
    print(f"Mean number of class per image: {classes.sum(dim=1).mean()}")
    
    n_classes = classes.shape[-1]
    n_activations = classes.shape[0]
    num_neurons = activations.shape[1]

    #activations = torch.nan_to_num(activations)
    classes = torch.nan_to_num(classes)
    
    print(f"Number of classes: {n_classes}")
    print(f"Number of neurons: {num_neurons}")
    print("Number of activations" ,len(activations))

    neuron_class_correlation = torch.corrcoef(torch.cat((activations.T, classes.T), dim=0))
    neuron_class_correlation = neuron_class_correlation[:num_neurons, num_neurons:]
    neuron_class_correlation = torch.nan_to_num(neuron_class_correlation)

    alive_neurons = neuron_class_correlation.abs().max(dim=1).values > 0.2
    l0 = activations.sum(dim=0)
    # sparse_neuron = l0 < 0.1 * l0.max()
    # alive_neurons = alive_neurons & sparse_neuron
    neuron_class_correlation = neuron_class_correlation[alive_neurons]
    print(f"Number of alive neurons: {neuron_class_correlation.shape[0]}")
    # get idx of alive neurons
    neuron_class_alive_idx = np.arange(len(alive_neurons))[alive_neurons.cpu().numpy()]

    # Compute the entropy score for each neuron considering the normalized (with the sum) correlation scores of the classes as the distribution
    neuron_class_entropy = neuron_class_correlation.abs() 
    neuron_class_entropy = neuron_class_entropy / (neuron_class_entropy.sum(dim=1, keepdim=True) + 1e-8)  # Normalize by the sum of the correlation scores
    neuron_class_entropy = torch.nan_to_num(neuron_class_entropy, nan=0.0)  # Replace NaN with 0.0
    assert torch.all(neuron_class_entropy >= 0), "All probabilities must be non-negative."
    assert torch.allclose(torch.sum(neuron_class_entropy, dim=1), torch.tensor(1.0)), "Probabilities must sum to 1."

    neuron_class_entropy = -torch.sum(neuron_class_entropy * torch.log(neuron_class_entropy + 1e-10), dim=1)    
    neuron_class_entropy = neuron_class_entropy.cpu().numpy()
    neuron_class_entropy_idx = np.argsort(neuron_class_entropy)
    neuron_class_entropy_idx = neuron_class_entropy_idx.copy()
    neuron_class_entropy = neuron_class_entropy[neuron_class_entropy_idx]
    neuron_class_correlation = neuron_class_correlation[neuron_class_entropy_idx, :]

    # Plot the neuron-class correlation matrix for the alive neurons and representing the correlation score by a color map
    neuron_class_correlation = neuron_class_correlation.cpu().numpy()
    neuron_class_correlation = pd.DataFrame(neuron_class_correlation, columns=labels_name)
    neuron_class_correlation['neuron_id'] = [f'{neuron_class_alive_idx[i]}' for i in neuron_class_entropy_idx]

    

    
    print(f"Top 3 correlated neurons per class and their entropy score:")
    for i in range(n_classes):
        top_neurons = neuron_class_correlation.iloc[:, i].nlargest(1)
        print(f"\nClass: {neuron_class_correlation.columns[i]}")
        for idx, score in top_neurons.items():
            print(f"Neuron: {neuron_class_correlation['neuron_id'][idx]}, Correlation: {score:.4f}, Entropy: {neuron_class_entropy[idx]:.4f}")


    # Plot the neuron-class correlation matrix using a heatmap seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Get shape
    num_neurons, num_classes = neuron_class_correlation.shape
    neuron_class_correlation = neuron_class_correlation.set_index('neuron_id')
    #neuron_class_correlation = neuron_class_correlation.drop(columns=['neuron_id'])

    # Use seaborn context for global scaling
    sns.set_context("notebook", font_scale=1.2)

    # Dynamically size figure based on data shape (no hard limits)
    fig_width = num_classes * 0.4
    fig_height = num_neurons * 0.3

    plt.figure(figsize=(fig_width, fig_height))

    # select 100 random indices from the neurons among the column of index of neuron_class_correlation    
    indices_random = np.random.choice(num_neurons, size=50, replace=False)

    # Create heatmap
    heatmap = sns.heatmap(
        neuron_class_correlation.iloc[indices_random,:],
        cmap='coolwarm',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.6}
    )

    # Adjust axes
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    plt.xticks(rotation=60, ha='left')
    plt.yticks(rotation=0)

    # Title and labels
    name = sae_name if args.sae_path is not None else model_name
    plt.title(f'Neuron-Class Correlation for:\n'
            f'{name}\n'
            f'on {dataset_name} ({args.split})\n'
            f'at the {args.hierarchy_level} hierarchy level',
            fontweight='bold',
            pad=20)

    plt.xlabel('Classes', fontweight='bold')
    plt.ylabel('Neurons', fontweight='bold')

    # Improve spacing
    plt.tight_layout()

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    filename = f'neuron_class_correlation_{args.dataset}_{args.split}_Layer{args.layer}_hierarchy{args.hierarchy_level}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()




if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)

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
    dataloader, labels = utils.get_dataset(args.dataset, batch_size, labels=True, shuffled_version=False)
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

    #mean center and normalize activations
    activations = activations - activations.mean(dim=0, keepdim=True)
    activations = F.normalize(activations, dim=0)


    print(f"Number of classes: {n_classes}")
    print(f"Number of neurons: {num_neurons}")
    print("Number of activations" ,len(activations))

    
    variances = torch.var(activations, dim=0)
    # Identify dead features with zero variance
    alive_features_indices = torch.where(variances != 0)[0]

    alive_activations = activations[:, alive_features_indices]
    alive_classes = classes[:, :]
    print(alive_activations.shape, alive_classes.shape)

    class_activation_means = []
    not_class_activation_means = []

    for c in range(n_classes):
        class_activations = alive_activations[alive_classes[:, c] == 1]
        
        not_class_activations = alive_activations[alive_classes[:, c] == 0]
        print(class_activations.shape, not_class_activations.shape)
        if class_activations.shape[0] > 0:
            class_mean = class_activations.mean(dim=0)
            class_activation_means.append(class_mean)
            not_class_mean = not_class_activations.mean(dim=0) if not_class_activations.shape[0] > 0 else torch.zeros_like(class_mean)
            not_class_activation_means.append(not_class_mean)

    class_activation_means = torch.stack(class_activation_means)
    not_class_activation_means = torch.stack(not_class_activation_means)
    monosemanticity_scores = torch.zeros((n_classes, class_activation_means.shape[1]), device=device)


    for neuron in range(len(monosemanticity_scores)):
        for c in range(n_classes):
            monosemanticity_scores[neuron, c] = class_activation_means[c, neuron]/(class_activation_means[c, :].sum() + 1e-8) - not_class_activation_means[c, neuron]/(not_class_activation_means[c, :].sum() + 1e-8)

    output_dir = os.path.join(config.RESULTS, args.sae_path.split('.')[0] if args.sae_path else model_name.lower())

    #plot histogram of monosemanticity scores
    monosemanticity_scores_flat = monosemanticity_scores.flatten()
    print(monosemanticity_scores_flat[:100])
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.hist(monosemanticity_scores_flat.cpu().numpy(), bins=100, alpha=0.7, color='blue')
    plt.title('Histogram of Monosemanticity Scores')
    plt.xlabel('Monosemanticity Score')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig(os.path.join(output_dir, f'monosemanticity_scores_histogram.png'))
    plt.close()

    monosemanticity_scores = monosemanticity_scores.flatten()
        
    #monosemanticity_scores = monosemanticity_scores.max(dim=1).values
    monosemanticity_scores = monosemanticity_scores.cpu().numpy()

    print(f"Monosemanticity scores computed for {num_neurons} neurons.")
    print(f"Max : {monosemanticity_scores.max()}")
    print(f"Min : {monosemanticity_scores.min()}")
    print(f"Mean : {monosemanticity_scores.mean()}")
    print(f"Std : {monosemanticity_scores.std()}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)

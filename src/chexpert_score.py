import torch
import os.path
import argparse
import os
import pandas as pd
import config
import utils
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer

METRICS = ['correlation', 'precision', 'recall', 'F1']

def get_args_parser():
    parser = argparse.ArgumentParser("Measure class correlation to neuron activations.", add_help=False)
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
    parser.add_argument('--neurons_to_print', type=int, required=False, help=f"The number of neuron's correlation scores that will be printed.", default=20)
    parser.add_argument('--l0_filter_min', type=int, required=False, help=f"The minimum l0 value to be considered a valid neuron in the computation of the scores.", default=15)
    parser.add_argument('--l0_filter_max', type=int, required=False, help=f"The maximum l0 value to be considered a valid neuron in the computation of the scores.", default=300)
    parser.add_argument('--percentile_top_act', type=float, required=False, help=f"The percentile of top activations you want for considering that a neuron fired.", default=0.30)
    parser.add_argument('--sort_by', type=str, required=False, help=f"Metric by which to sort the printed neuron's scores. Available metrics : [{', '.join(METRICS)} ]", default='precision')
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="trainval")   
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)
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
    
    print(f"Mean number of class per image: {classes.sum(dim=1).mean()}")
    n_classes = classes.shape[-1]
    n_activations = classes.shape[0]
    num_neurons = activations.shape[1]


    # Scale to 0-1 per neuron
    activations_raw = activations.clone()
    percentile = args.percentile_top_act
    thresholds = []
    for i in range(activations.shape[1]):
        # Get positive activations for neuron i
        pos_acts = activations[:, i][activations[:, i] > 1e-7]
        if len(pos_acts) == 0:
            thresholds.append(torch.tensor(float('inf'), device=device))  # No positive activations
            continue

        k = max(1, int((1. - percentile) * len(pos_acts)))  # kth largest to keep
        # Note: kthvalue is 1-based index
        threshold = torch.kthvalue(pos_acts, k=k).values
        thresholds.append(threshold)

    thresholds = torch.stack(thresholds)  # shape: [num_neurons]

    # Broadcast threshold per column
    thresholds = thresholds.unsqueeze(0)  # shape: [1, num_neurons]
    mask = activations > thresholds  # shape: [batch_size, num_neurons]
    activations = torch.where(mask, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))

    #activations = torch.nan_to_num(activations)
    classes = torch.nan_to_num(classes)
    

    print("Number of activations" ,len(activations))

    neuron_class_correlation = torch.corrcoef(torch.cat((activations_raw.T, classes.T), dim=0))
    neuron_class_correlation = neuron_class_correlation[:num_neurons, num_neurons:]
    os.makedirs(output_dir, exist_ok=True)
    torch.save(neuron_class_correlation, os.path.join(output_dir, f"neuron_class_correlation_{args.dataset}_{args.split}.pth"))

    neuron_class_prec = ((activations/activations.sum(dim=0)).T)@(classes)
    neuron_class_rec = ((activations).T)@(classes/classes.sum(dim=0, keepdim=True))
    neuron_class_f1 = 2 * (neuron_class_prec * neuron_class_rec) / (neuron_class_prec + neuron_class_rec + 1e-8)
    l0 = activations.sum(dim=0)
    l0 = l0 // args.percentile_top_act  
    
    

    nan_mask = l0 == 0
    dead_neurons = nan_mask.sum().item()

    print(f"Number of classes: {n_classes}")
    print(f"Dead neurons:", dead_neurons)
    print(f"Total neurons:", num_neurons)

    # Filter out NaNs
    
    mask = (l0 > args.l0_filter_min)*(l0 < args.l0_filter_max)
    valid_indices = torch.where(mask)[0]
    valid_scores_prec = neuron_class_prec[mask,:]
    valid_scores_rec = torch.nan_to_num(neuron_class_rec[mask,:])
    valid_scores_f1 = neuron_class_f1[mask,:]
    valid_scores_f1 = torch.nan_to_num(valid_scores_f1)
    valid_scores_prec = torch.nan_to_num(valid_scores_prec)
    valid_scores_corr = torch.nan_to_num(neuron_class_correlation[mask,:])

    

    metric_scores = {
            'correlation': valid_scores_corr,
            'precision': valid_scores_prec,
            'recall': valid_scores_rec,
            'F1': valid_scores_f1
        }
    
    specific_neurons_to_print_results = [7745,1291,3324,7036,820,7205,1774,4186]
    for neuron in specific_neurons_to_print_results:
        if neuron in valid_indices:
            idx = valid_indices.tolist().index(neuron)
            print(f"Neuron {neuron} - l0: {l0[neuron].item()}")
            class_idx = torch.argmax(valid_scores_f1[idx])
            print(f"  Class: {labels_name[class_idx]} - F1: {valid_scores_f1[idx, class_idx].item():.3f} - Prec.: {valid_scores_prec[idx, class_idx].item():.3f} - Rec.: {valid_scores_rec[idx, class_idx].item():.3f} - Correlation: {valid_scores_corr[idx, class_idx].item():.3f}")

    os.makedirs(output_dir, exist_ok=True)
    torch.save(metric_scores[args.sort_by], os.path.join(output_dir, f"neuronclass_score_{args.dataset}_{args.split}_{args.sort_by}_l0min{args.l0_filter_min}_l0max{args.l0_filter_max}_percentile{args.percentile_top_act}.pth"))

    # Get top 10 highest 
    top_10_values, top_10_indices = torch.topk(metric_scores[args.sort_by], 5, dim=1)

    top_10_values, top_10_indices = top_10_values.cpu(), top_10_indices.cpu()
    
    sorted_values, sorted_indices = torch.sort(top_10_values[:,0], dim=0, descending=True)

    nb_of_printed_neurons = min(args.neurons_to_print, len(sorted_indices))

    # Create row labels: "Top 1", "Top 2", ..., "Top k"
    row_labels = [f"Top {i+1} class" for i in range(5)]
    # Create column labels: "Class 1", "Class 2", ..., "Class n_classes"
    column_labels = [f"Neuron {valid_indices[sorted_indices[j]]} - l0 : {l0[valid_indices[sorted_indices[j]]].item()}" for j in range(nb_of_printed_neurons)]
    # Create a DataFrame with the top-k output
    df = pd.DataFrame(top_10_values[sorted_indices[:nb_of_printed_neurons],:].T, index=row_labels, columns=column_labels, dtype=str)
    for row in range(len(df)):
        for col in range(len(df.columns)):
            df.iloc[row, col] = f"Cor {float(valid_scores_corr[sorted_indices[col],top_10_indices[sorted_indices[col]][row]]):.3f} - F1 {float(valid_scores_f1[sorted_indices[col],top_10_indices[sorted_indices[col]][row]])*100:.1f}% - Prec. {float(valid_scores_prec[sorted_indices[col],top_10_indices[sorted_indices[col]][row]])*100:.1f}% - Rec. {float(valid_scores_rec[sorted_indices[col],top_10_indices[sorted_indices[col]][row]])*100:.1f}% - class : {labels_name[top_10_indices[sorted_indices[col]][row]]}"            

    top_df = df.copy()

    

    # Print results
    print("Top 5 correlated neurons:")
    print(top_df)


    # Save to file
    output_path = os.path.join(output_dir, f"neuronclass_stats_{args.dataset}_{args.split}_{args.sort_by}_l0min{args.l0_filter_min}_l0max{args.l0_filter_max}_percentile{args.percentile_top_act}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top_df.to_csv(output_path, index=False)
    # with open(output_path, "w") as file:
    #     file.write(f"Dead neurons: {dead_neurons}\n")
    #     file.write(f"Total neurons: {num_neurons}\n\n")

    #     file.write("Top 5 most class correlated neurons:\n")
    #     file.write(top_df.to())
    #     file.write("\n\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)

import torch
import os.path
import argparse
import os
import pandas as pd
import config
import utils
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, numbers
from openpyxl.utils.dataframe import dataframe_to_rows

def rank_normalize(scores: torch.Tensor) -> torch.Tensor:
    """
    Perform rank-based normalization per class (column).
    Higher score = better rank. Normalized to [0, 1].
    """
    num_models, num_classes = scores.shape
    # Initialize normalized rank tensor
    norm_ranks = torch.zeros_like(scores, dtype=torch.float32)

    for c in range(num_classes):
        # Extract scores for class c
        class_scores = scores[:, c]

        # Sort indices descending (higher score = better rank)
        sorted_indices = torch.argsort(class_scores, descending=True)
        
        # Create ranks: best = 0, worst = num_models - 1
        ranks = torch.zeros_like(sorted_indices, dtype=torch.float32)
        ranks[sorted_indices] = torch.arange(num_models, dtype=torch.float32)

        # Normalize: best = 1.0, worst = 0.0
        norm_ranks[:, c] = 1.0 - ranks / (num_models - 1)

    return norm_ranks


def get_args_parser():
    parser = argparse.ArgumentParser("Measure class correlation via weighted class appartenance.", add_help=False)
    parser.add_argument('--folder_path', type=str, required=False, default=None, help=f'The path of the folder of the SAE Features starting from {config.FEATURES}.')
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="trainval")   
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation (default: cuda)')
    parser.add_argument('--hierarchy_level', type=str, default='both', choices=['low', 'high', 'both'], help='Hierarchy level to use for CheXpert labels (default: both)')
    return parser

def main(args):
    import wandb

    api = wandb.Api()
    wandb_project = "SAE_MedCLIP_Chexpert_Standard_sweep_NewScale"
    wandb_entity = ""
    runs = api.runs(f"{wandb_entity}/{wandb_project}")

        
    device=args.device

    dataset_name = args.dataset
    dataloader, labels = utils.get_dataset(args.dataset, 512, labels=True, shuffled_version=True)
    dataloader = dataloader[args.split]
    labels = labels[args.split]
    folder_path = os.path.join(config.FEATURES,args.folder_path) if args.folder_path else config.FEATURES

    if args.split == 'trainval':
        classes = labels.dataset.dataset.labels[labels.dataset.indices].to(device)
        labels_name = labels.dataset.dataset.labels_name
    else:
        classes = labels.dataset.labels.to(device)
        labels_name = labels.dataset.labels_name
    
    labels_selected = [0,2,4,5,7,9,10,13]
    labels_name = labels_name[labels_selected]
    classes = classes[:, labels_selected]

    if args.hierarchy_level == 'low':
        idx = [labels_name.index(label) for label in config.CHEXPERT_LOW_LABELS if label in labels_name]
        classes = classes[:, idx]
        labels_name = [labels_name[i] for i in idx]
    elif args.hierarchy_level == 'high':
        idx = [labels_name.index(label) for label in config.CHEXPERT_HIGH_LABELS if label in labels_name]
        classes = classes[:, idx]
        labels_name = [labels_name[i] for i in idx]
    classes = torch.nan_to_num(classes)
    output_dir = os.path.join(config.RESULTS)
    print(f"Mean number of class per image: {classes.sum(dim=1).mean()}")

    sae_best_neuron_list = []
    sae_mean_neuron_list = []
    filenames = []
    losses = []
    dead_feature_rates = []
    top_neurons_mean = np.array([])
    top_neurons_entropy_mean = np.array([])
    neuron_class_sae_corr = None
    neuron_class_sae_entr = None
    saved_runs = []
    fve = []
    l0_rate = []
    run = None
    for file in os.listdir(folder_path):
        
        if os.path.isdir(os.path.join(folder_path, file)): continue
        activations = torch.load(os.path.join(folder_path, file), map_location=device)
        

        
        file_name = ".".join(file.split('.')[:-1])
        
        n_classes = classes.shape[-1]
        n_activations = classes.shape[0]
        num_neurons = activations.shape[1]

        alive_features_count = (activations != 0).sum(dim=0)
        dead_feature_rates.append((alive_features_count < 1e-7).sum()/num_neurons)

        neuron_class_correlation = torch.corrcoef(torch.cat((activations.T, classes.T), dim=0))
        
        neuron_class_correlation = neuron_class_correlation[:num_neurons, num_neurons:]
        alive_neurons = neuron_class_correlation.abs().max(dim=1).values > 0.2
        if alive_neurons.sum() == 0: continue
        for run_ in runs:
            if run_.name == file_name:
                run = run_
                saved_runs.append(run)

        neuron_class_correlation = neuron_class_correlation[alive_neurons]
        # Compute the entropy score for each neuron considering the normalized (with the sum) correlation scores of the classes as the distribution
        #neuron_class_entropy = neuron_class_correlation - neuron_class_correlation.min(dim=1).values.unsqueeze(1)  # Shift the minimum value to 0
        neuron_class_entropy = neuron_class_correlation.abs()
        BCEloss = torch.nn.BCELoss()
        target = torch.zeros_like(neuron_class_correlation)
        maximums = neuron_class_correlation.argmax(dim=1)
        for i in range(len(maximums)):
            target[i, maximums[i]] = 1
        losses.append(BCEloss(neuron_class_correlation.abs(), target).item())
        neuron_class_entropy = neuron_class_entropy / (neuron_class_entropy.sum(dim=1, keepdim=True) + 1e-8)  # Normalize by the sum of the correlation scores
        neuron_class_entropy = torch.nan_to_num(neuron_class_entropy, nan=0.0)  # Replace NaN with 0.0
        assert torch.all(neuron_class_entropy >= 0), "All probabilities must be non-negative."
        assert torch.allclose(torch.sum(neuron_class_entropy, dim=1), torch.tensor(1.0)), "Probabilities must sum to 1."

        neuron_class_entropy = -torch.sum(neuron_class_entropy * torch.log(neuron_class_entropy + 1e-10), dim=1)    
        neuron_class_entropy = neuron_class_entropy
        
        top_neurons, top_neurons_idx = neuron_class_correlation.max(dim=0)
        
        top_neurons_entropy = neuron_class_entropy[top_neurons_idx]

        if neuron_class_sae_corr == None: 
            neuron_class_sae_corr = top_neurons.unsqueeze(0)
            neuron_class_sae_entr = top_neurons_entropy.unsqueeze(0)
        else:
            neuron_class_sae_corr = torch.cat((neuron_class_sae_corr, top_neurons.unsqueeze(0)),0)
            neuron_class_sae_entr = torch.cat((neuron_class_sae_entr, top_neurons_entropy.unsqueeze(0)),0)

        top_neurons = top_neurons.cpu().numpy()
        top_neurons_entropy = top_neurons_entropy.cpu().numpy()
        top_neurons_str = []
        top_neurons_mean = np.append(top_neurons_mean,[top_neurons.mean()])
        top_neurons_entropy_mean = np.append(top_neurons_entropy_mean,[top_neurons_entropy.mean()])
        sae_mean_neuron_list.append(f"{top_neurons.mean():.2f} ({top_neurons_entropy.mean():.2f})")
        for i in range(len(top_neurons_entropy)):
            top_neurons_str.append(f"{top_neurons[i]:.2f} ({top_neurons_entropy[i]:.2f})")
        sae_best_neuron_list.append(top_neurons_str)
        l0_rate.append(((activations.to(dtype=torch.float) != 0).sum(dim=-1).float().mean()).item() / num_neurons)
        filenames.append(file_name)
        if run:
            run.summary["external/entropy_mean"] = top_neurons_entropy.mean()
            run.summary["external/BCE_loss"] = losses[-1]
            run.summary["external/correlation_mean"] = top_neurons.mean()
            run.summary["eval/l0_rate"] = run.summary["eval/l0"]/run.config["dict_size"]
            run.summary.update()
            
            fve.append(run.summary["eval/frac_variance_explained"])
        else:
            fve.append(0.0)


    norm_neuron_class_sae_corr_mean = rank_normalize(neuron_class_sae_corr.cpu()).mean(dim=1)
    norm_neuron_class_sae_entr_mean = rank_normalize(neuron_class_sae_entr.cpu()).mean(dim=1)
    for i, run in enumerate(saved_runs):
        run.summary["external/ranked_norm_entropy_mean"] = norm_neuron_class_sae_entr_mean[i]
        run.summary["external/ranked_norm_correlation_mean"] = norm_neuron_class_sae_corr_mean[i]
        run.summary.update()

    losses = np.array(losses)
    print("Min entropy mean :", top_neurons_entropy_mean.min())
    print("Min loss :", losses.min())
    print("Max correlation mean :", top_neurons_mean.max())
    indices = norm_neuron_class_sae_entr_mean.argsort()
    results = pd.DataFrame([sae_best_neuron_list[i] for i in indices], index=[filenames[i] for i in indices], columns=labels_name)
    results = results.round(3)
    results = results.T
    wb = Workbook()
    ws = wb.active
    
    for r in dataframe_to_rows(results, index=True, header=True):
        ws.append(r)
    
    bold_font = Font(bold=True)

    for row in ws.iter_rows(min_row=3, max_row=len(labels_name)+2, min_col=2, max_col=len(filenames)+1):
        max_value = max([float(cell.value.split(' ')[0]) for cell in row])
        for cell in row:
            if float(cell.value.split(' ')[0]) == max_value:
                cell.font = bold_font
     
    # Add a mean row at the bottom
    mean_row = ['Mean'] + [sae_mean_neuron_list[i] for i in indices]
    ws.append(mean_row)
    mean_ranked_row = ['Mean Ranked Norm'] + [f"{norm_neuron_class_sae_corr_mean[i]:.3f} ({norm_neuron_class_sae_entr_mean[i]:.3f})" for i in indices]
    ws.append(mean_ranked_row)
    loss_row = ['BCELoss'] + [f"{losses[i]:.3f}" for i in indices]
    ws.append(loss_row)
    l0_row = ['L0 Rate'] + [f"{l0_rate[i]:.3f}" for i in indices]
    ws.append(l0_row)
    fve_row = ['Fraction Variance Explained'] + [f"{fve[i]:.3f}" for i in indices]
    ws.append(fve_row)
    dead_feature_row = ['Dead Feature Rate'] + [f"{dead_feature_rates[i]*100:.1f}%" for i in indices]
    ws.append(dead_feature_row)

    # # Plot a table with in bold the highest correlation for each class in latex format
    # for label in labels_name:
    #     results[label] = results[label].apply(lambda x: f"\\textbf{{{x:.2f}}}" if x == results[label].max() else f"{x:.2f}")
    # results = results.T
    #results.to_excel(os.path.join(output_dir, f"neuron_class_correlation_{dataset_name}_{args.split}.xlsx"))
    wb.save(os.path.join(output_dir, f"neuron_class_correlation_{dataset_name}_{args.split}.xlsx"))
    print(f"Results saved to {os.path.join(output_dir, f'neuron_class_correlation_{dataset_name}_{args.split}.xlsx')}")



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)

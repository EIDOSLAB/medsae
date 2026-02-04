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


def sigmoid(x):
    return 1 / (1 + t.exp(-x))

def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = t.clamp(y_pred, eps, 1 - eps)
    return -(y_true * t.log(y_pred) + (1 - y_true) * t.log(1 - y_pred)).mean(dim=0)

@t.no_grad()
def batched_newton_raphson(activation, y, max_iter=20, tol=1e-6):
    """
    Train logistic probes in batch over latents using Newton-Raphson.

    activation: (N, D) t.float16 or float32
    y: (N,) binary labels (0 or 1)
    Returns:
        losses: (D,)
        w: (D,)
        b: (D,)
    """
    N, D = activation.shape
    device = activation.device
    dtype = activation.dtype

    x = activation  # (N, D)
    y = y.unsqueeze(1).expand(-1, D)  # (N, D)

    w = t.zeros(D, device=device, dtype=dtype)
    b = t.zeros(D, device=device, dtype=dtype)

    for _ in range(max_iter):
        logits = x * w + b  # (N, D)
        probs = sigmoid(logits)

        error = probs - y  # (N, D)

        # Gradients
        grad_w = (error * x).mean(dim=0)
        grad_b = error.mean(dim=0)

        # Hessians
        s = probs * (1 - probs)  # (N, D)
        hess_w = (s * x * x).mean(dim=0).clamp(min=1e-6)
        hess_b = s.mean(dim=0).clamp(min=1e-6)

        # Update
        delta_w = grad_w / hess_w
        delta_b = grad_b / hess_b

        w -= delta_w
        b -= delta_b

        if t.max(t.abs(delta_w)) < tol and t.max(t.abs(delta_b)) < tol:
            break

    # Final loss
    final_logits = x * w + b
    final_probs = sigmoid(final_logits)
    loss = binary_cross_entropy(y, final_probs)
    return loss, w, b

def find_best_latent_probe_optimized(activation, labels, chunk_size=1024, n_samples=200,max_iter=20, use_fp16=True):
    """
    Efficiently train 1D logistic probes for each latent and class.

    Returns:
        best_loss, best_class, best_latent, (w, b)
    """
    if use_fp16:
        activation = activation.half()
        labels = labels.half()
    else:
        activation = activation.float()
        labels = labels.float()

    N, D = activation.shape
    C = labels.shape[1]
    device = activation.device

    best_loss_list = []
    best_class_list = []
    best_latent_list = []
    best_params_list = []

    for class_idx in tqdm(range(C), desc="🔍 Classes"):
        class_indices = t.where(labels[:, class_idx] == 1)[0]
        sampled_class_indices = class_indices[t.randperm(len(class_indices))[:n_samples]]
        other_indices = t.where(labels[:, class_idx] == 0)[0]
        sampled_other_indices = other_indices[t.randperm(len(other_indices))[:n_samples]]
        sampled_indices = t.cat([sampled_class_indices, sampled_other_indices])[t.randperm(2*n_samples)]
        y = labels[sampled_indices, class_idx]  # (N,)
        x = activation[sampled_indices]
        
        best_loss = float('inf')
        best_class = -1
        best_latent = -1
        best_params = (0.0, 0.0)
        for i in range(0, D, chunk_size):
            z_chunk = x[:, i:i+chunk_size]  # (N, chunk)
            losses, ws, bs = batched_newton_raphson(z_chunk, y, max_iter=max_iter)
            min_loss, min_idx = t.min(losses, dim=0)
            if min_loss.item() < best_loss:
                best_loss = min_loss.item()
                best_class = class_idx
                best_latent = i + min_idx.item()
                best_params = (ws[min_idx].item(), bs[min_idx].item())
        best_loss_list.append(best_loss)
        best_class_list.append(best_class)
        best_latent_list.append(best_latent)
        best_params_list.append(best_params)

    print(f"\n🏁 Best Probe → Class: {best_class}, Latent: {best_latent}, Loss: {best_loss:.6f}")
    return best_loss_list, best_class_list, best_latent_list, best_params_list

def evaluation(params, latents, activations, labels):
    # Initialize lists to store the counts and metrics
    FP = [0 for _ in latents]
    TP = [0 for _ in latents]
    FN = [0 for _ in latents]
    TN = [0 for _ in latents]

    precision = [0.0 for _ in latents]
    recall = [0.0 for _ in latents]
    f1 = [0.0 for _ in latents]

    for idx, latent in enumerate(latents):
        # Compute logits
        z = activations[:, latent] * params[idx][0] + params[idx][1]

        # Convert logits to probabilities using sigmoid
        probs = sigmoid(z)

        # Convert probabilities to binary predictions
        predictions = (probs > 0.5).float()
        

        # Compute TP, FP, FN, TN
        TP[idx] = t.sum((predictions == 1) & (labels[:, idx] == 1)).item()
        FP[idx] = t.sum((predictions == 1) & (labels[:, idx] == 0)).item()
        FN[idx] = t.sum((predictions == 0) & (labels[:, idx] == 1)).item()
        TN[idx] = t.sum((predictions == 0) & (labels[:, idx] == 0)).item()
        

        # Calculate Precision, Recall, and F1 score
        precision[idx] = TP[idx] / (TP[idx] + FP[idx]) if (TP[idx] + FP[idx]) > 0 else 0.0
        recall[idx] = TP[idx] / (TP[idx] + FN[idx]) if (TP[idx] + FN[idx]) > 0 else 0.0
        f1[idx] = 2 * (precision[idx] * recall[idx]) / (precision[idx] + recall[idx]) if (precision[idx] + recall[idx]) > 0 else 0.0
    print(TP)
    print(FP)
    print(FN)
    print(TN)

    return precision, recall, f1


def main(args):
    model_name = args.model_name
    batch_size = args.batch_size
    device=args.device
    dataset_name = args.dataset
    dataloader, labels = utils.get_dataset(args.dataset, batch_size, labels=True, shuffled_version=False)
    dataloader = dataloader[args.split]
    labels = labels[args.split]

    #Model import
    print(f"Importing model {model_name}...")
    model, processor = utils.get_model(model_name)

    #Buffer set-up
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

        for i, _ in tqdm(enumerate(feature_buffer), total=len(feature_buffer), desc="Computing SAE Features" if activation_buffer.load_activations else "Computing SAE Features and dataset activations"):
            pass
        activations = feature_buffer.features
        output_dir = os.path.join(config.RESULTS, args.sae_path.split('.')[0])
    else:   
        for i, _ in tqdm(enumerate(activation_buffer), total=len(activation_buffer), desc="Computing dataset activations"):
            pass
        activations = activation_buffer.activations.to(device)
        output_dir = os.path.join(config.RESULTS, model_name.lower())


    if args.split == 'trainval':
        classes = labels.dataset.dataset.labels[labels.dataset.indices].to(device)
        labels_name = labels.dataset.dataset.labels_name
    else:
        classes = labels.dataset.labels.to(device)
        labels_name = labels.dataset.labels_name

    n_classes = classes.shape[-1]
    n_activations = classes.shape[0]
    num_neurons = activations.shape[1]
    print(f"Number of classes: {n_classes}")
    print(f"Number of neurons: {num_neurons}")
    print("Number of activations" ,len(activations))

    best_loss, best_label, best_latent, best_params = find_best_latent_probe_optimized(activations, classes, chunk_size=8192, max_iter=50)

    precision, recall, f1 = evaluation(best_params,best_latent,activations,classes)
    for i in range(len(precision)):
        print(f"label : {labels_name[i]}, neuron : {best_latent[i]}, precision : {precision[i]}, recall : {recall[i]}, F1 : {f1[i]}")
        print(f"Class rate : {(classes[:,i].sum()/n_activations)*100:3f}")
        print("---------------")


    file = pd.DataFrame({"Class": labels_name,"Neuron":best_latent,"Loss":best_loss, "F1": f1, "Precision":precision, "Recall":recall})
    saving_path = os.path.join(output_dir,"1d_probe.csv")
    file.to_csv(saving_path, index=False)
    print(f"CSV file saved at : {saving_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that trains a binary 1-d probe on each dimension of the representation for every class of the dataset (see 1d-probe method).")
    parser.add_argument('--sae_path', type=str, required=False, default=None, help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the vocab activations', default=1024)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features. If not provided it will automatically save the sae's features to {config.FEATURES}.", default=None)
    parser.add_argument('--load_features_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the sae's features. If not provided it will automatically check if features can be loaded from {config.FEATURES}.", default=None)
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="")   
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)

    args = parser.parse_args()
    main(args)



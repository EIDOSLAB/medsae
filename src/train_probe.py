from dictionary_learning.dictionary import AutoEncoder, AutoEncoderNew
from torch.utils.data import Dataset
import torch as t
from torcheval.metrics import MultilabelAccuracy
import torchvision
import data_preprocess
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
import clip
from tqdm.auto import tqdm
import argparse
import config
import os
import utils
import pandas as pd
import math
import torchmetrics
from PIL import Image
import wandb
import numpy as np
from contextlib import nullcontext
import torch
import torch.nn.functional as F

def manual_cross_entropy_loss(logits, targets):
    """
    Computes the cross entropy loss manually.
    
    Args:
        logits: Tensor of shape [batch_size, num_classes] – raw outputs from the model
        targets: Tensor of shape [batch_size] – class indices (not one-hot)
    
    Returns:
        Scalar tensor: average cross-entropy loss over the batch
    """
    log_probs = F.log_softmax(logits, dim=1)          # Apply log softmax
    nll_loss = -log_probs[range(len(targets)), targets]  # Negative log likelihood for correct class
    return nll_loss.mean()


def main(args):
    sae_path = os.path.join(config.SAE_CKPTS,args.sae_path)

    sae = AutoEncoder.from_pretrained(sae_path).to(args.device)
    #sae = AutoEncoderNew.from_pretrained(sae_path).to(args.device)
    #sae = BatchTopKSAE.from_pretrained(sae_path).to(args.device)

    activation_dim = sae.activation_dim
    dict_size = sae.dict_size
    expansion_factor = dict_size/activation_dim

    if args.use_wandb: wandb.init(project=args.wandb_project , entity=args.wandb_entity,name=f"{args.sae_path}_{args.dataset}_{args.lr}_{args.epochs}", config=vars(args))

    batch_size = args.batch_size
    model_name = args.model_name
    device=args.device
    dataloader, labels = utils.get_dataset(args.dataset, batch_size, labels=True)
    train_labels, val_labels = labels['train'], labels['val']
    

    #Model import
    print(f"Importing model {model_name}...")
    model, processor = utils.get_model(model_name)

    #Buffer set-up
    print(f"Setting up activation buffer...")
    train_activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader['train'],
        modality='img',
        model=model,
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        load_activations_path=os.path.join(config.ACTIVATIONS,args.load_activations_path+"_train.pt") if args.load_activations_path else None,
        save_activations_path=os.path.join(config.ACTIVATIONS,args.save_activations_path+"_train.pt") if args.save_activations_path else None,
        shuffle=False,
    )


    val_activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader['val'],
        modality='img',
        model=model,
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        load_activations_path=os.path.join(config.ACTIVATIONS,args.load_activations_path+"_val.pt") if args.load_activations_path else None,
        save_activations_path=os.path.join(config.ACTIVATIONS,args.save_activations_path+"_val.pt") if args.save_activations_path else None
    )

    print(f"Setting up feature buffer...")
    train_feature_buffer = FeatureBuffer(
        activation_buffer=train_activation_buffer,
        sae=sae,
        out_batch_size=batch_size,
        device=device,
        load_features_path=os.path.join(config.FEATURES,args.load_features_path+"_train.pt") if args.load_features_path else None,
        save_features_path=os.path.join(config.FEATURES,args.save_features_path+"_train.pt") if args.save_features_path else None
    )

    val_feature_buffer = FeatureBuffer(
        activation_buffer=val_activation_buffer,
        sae=sae,
        out_batch_size=batch_size,
        device=device,
        load_features_path=os.path.join(config.FEATURES,args.load_features_path+"_val.pt") if args.load_features_path else None,
        save_features_path=os.path.join(config.FEATURES,args.save_features_path+"_val.pt") if args.save_features_path else None
    )

    if args.dataset == 'chexpert':
        num_classes = 14
        print(f"Number of classes: {num_classes}")
        classification_loss_fn = t.nn.BCEWithLogitsLoss()
    else:
        num_classes = args.nclasses
        classification_loss_fn = t.nn.CrossEntropyLoss()
    
    sparsity_loss_fn = t.nn.L1Loss() if args.sparsity_loss else None

    total = len(train_feature_buffer)
    total_val = len(val_feature_buffer)


    
    model = t.nn.Linear(
            dict_size, num_classes, bias=False)

    
    model = model.train().to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr)
    
    print("Precomputing features...")
    train_feature_buffer.precompute_features()
    val_feature_buffer.precompute_features()
    val_data = val_feature_buffer.features
    train_data = train_feature_buffer.features
    train_labels = t.concat(list(iter(train_labels)))
    val_labels = t.concat(list(iter(val_labels)))
    train_buffer = [(x,y) for (x,y) in zip(train_data, train_labels)]
    val_buffer = [(x,y) for (x,y) in zip(val_data, val_labels)]

    dataloader = t.utils.data.DataLoader(train_buffer, batch_size=batch_size, shuffle=True)
    val_dataloader = t.utils.data.DataLoader(val_buffer, batch_size=batch_size, shuffle=False)
    

    #with autocast_context:
    for e in range(args.epochs):
        model.train()
        total_loss = 0
        total_classification_loss = 0
        total_sparsity_loss = 0
        total_batches = 0
        for batch_idx, (train_X, train_y) in enumerate(tqdm(dataloader, total=total, desc=f"Epoch {e+1} :")):
            # if batch_idx == 0:
            #     if e > 0: assert t.allclose(previous_tensor, train_X)
            #     previous_tensor = train_X
            total_batches += 1
            model.zero_grad()
            train_X = train_X.to(args.device).to(dtype=model.weight.dtype)
            train_y = train_y.float().to(args.device)
            out = model(train_X)
            
            classification_loss = classification_loss_fn(out, train_y)
            
            if sparsity_loss_fn is not None:
                sparsity_loss = sparsity_loss_fn(
                    model.weight.flatten(), t.zeros_like(model.weight.flatten()))
                total_sparsity_loss += sparsity_loss.item()
            else:
                sparsity_loss = 0
            total_classification_loss += classification_loss.item()
            loss = classification_loss + args.sparsity_loss_lambda*sparsity_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()  
            

        avg_loss = total_loss/total_batches
        avg_class_loss = total_classification_loss/total_batches
        avg_sparse_loss = total_sparsity_loss/total_batches
        print(f"Epoch: {e+1}, Training Loss: {avg_loss}, Classification Loss: {avg_class_loss}, Sparsity Loss: {avg_sparse_loss}")



        accuracy_top1 = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro").to(device)
        accuracy_labels = MultilabelAccuracy(criteria="overlap").to(device)

        val_total_loss = 0
        val_total_classification_loss = 0
        val_total_sparsity_loss = 0
        val_total_batches = 0
        if (e + 1) % args.val_freq == 0:
            with t.no_grad():
                model.eval()
                for batch_idx, (eval_X, eval_y) in enumerate(tqdm(val_dataloader, total=total_val, desc="Evaluating ...")):
                    val_total_batches += 1
                    eval_X = eval_X.to(dtype=model.weight.dtype).to(args.device)
                    eval_y = eval_y.float().to(args.device)
                    out = model(eval_X)
                    accuracy_top1.update(out, eval_y)
                    accuracy_labels.update(out, eval_y)
                    val_classification_loss = classification_loss_fn(out, eval_y)
                    if sparsity_loss_fn is not None:
                        val_sparsity_loss = sparsity_loss_fn(
                            model.weight.flatten(), t.zeros_like(model.weight.flatten()))
                        val_total_sparsity_loss += val_sparsity_loss.item()
                    else:
                        val_sparsity_loss = 0
                    val_total_classification_loss += val_classification_loss.item()
                    val_loss = val_classification_loss + args.sparsity_loss_lambda*val_sparsity_loss
                    val_total_loss += val_loss.item()
                
                acc_top1 = accuracy_top1.compute()
                acc_labels = accuracy_labels.compute()
                avg_val_loss = val_total_loss/val_total_batches
                avg_val_class_loss = val_total_classification_loss/val_total_batches
                avg_val_sparse_loss = val_total_sparsity_loss/val_total_batches
                print(f"Validation - Top1 Accuracy: {acc_top1}, Labels accuracy: {acc_labels}, Loss: {avg_val_loss}, Classification Loss: {avg_val_class_loss}, Sparsity Loss: {avg_val_sparse_loss}")

            if args.use_wandb:   
                wandb.log({
                    "val/accuracy": acc_top1,
                    "val/label_accuracy": acc_labels,
                    "val/loss": avg_val_loss,
                    "val/classification_loss": avg_val_class_loss,
                    "val/sparsity_loss": avg_val_sparse_loss
                })

        if args.use_wandb:
            wandb.log({
                    "train/loss": avg_loss,
                    "train/classification_loss": avg_class_loss,
                    "train/sparsity_loss": avg_sparse_loss
                })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that plots the top activating images corresponding to the SAE's features.")
    parser.add_argument('--sae_path', type=str, required=False, default="ae.pt", help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the vocab activations', default=1024)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the folder of the training and val model's activation files starting from {config.ACTIVATIONS}. The file names must finish by _train.pt and _val.pt", default=None)
    parser.add_argument('--load_features_path', type=str, required=False, help=f"[Optional] Path of the folder of the training and val sae's feature files starting from {config.FEATURES}. The file names must be train.pt and val.pt", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features starting from {config.FEATURES}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations starting from {config.ACTIVATIONS}.", default=None)
    parser.add_argument("--nclasses", type=int, default=1000, help="number of classes in the probe dataset")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the probe.")
    parser.add_argument("--sparsity_loss", action="store_true", help="Whether to apply sparsity loss.")
    parser.add_argument("--sparsity_loss_lambda", type=float, default=1e-3, help="Lambda for sparsity loss.")
    parser.add_argument("--val_freq", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--use_wandb", default=False)
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="Probe_Training")
    args = parser.parse_args()
    main(args)
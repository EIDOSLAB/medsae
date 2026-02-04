

from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainer_config import StandardTrainerConfig
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import CLIPActivationBuffer

from dictionary_learning.training import trainSAE
import config

import data_preprocess
import clip

import torchvision
import torch
import utils

from dataclasses import asdict


import os
#CONFIG
batch_size = 1024
expansion_factor = 8
activation_dim = 512
n_epochs = 200
decay_start = None
device = 'cuda'
layer = ''
model_name = "MedCLIP-RN50"
submodule_name = ''
learning_rate = 3e-4
dict_size = expansion_factor*activation_dim
seed = None
l1_penalty = 3e-5
resample_frq = 0.1
warmup_frq = 0.00
sparsity_warmup_frq = 5*warmup_frq
use_wandb = True
wandb_project = "SAE_MedCLIP"
wandb_entity = ""
save_steps = [2, 5, 10, 20, 50, 100, 150] 
save_dir = config.SAE_CKPTS+f"/{model_name}/lr{learning_rate}_l1{l1_penalty}_bs{batch_size}_exp{expansion_factor}_n_epochs{n_epochs}_rf{resample_frq}_wf{warmup_frq}_swf{sparsity_warmup_frq}/"
log_steps = 2 # 
eval_steps = 10 # Steps = batches




#Dataset preprocess
dataloader = utils.get_dataset('chexpert', batch_size)

resample_steps = len(dataloader['train'])/resample_frq
print(resample_steps)
steps = int(len(dataloader['train'])*n_epochs)
warmup_steps = warmup_frq*steps
sparsity_warmup_steps = sparsity_warmup_frq*warmup_steps




#Model import
print(f"Importing model {model_name}...")
model, processor = utils.get_model(model_name)

#Buffer set-up
print(f"Setting up activation buffer... Nb of training batches : {len(dataloader['train'])}, Nb of validation batches : {len(dataloader['val'])}")
train_activation_buffer = CLIPActivationBuffer.get(
    model_name,
    dataloader=dataloader['train'],
    modality='img',
    model=model,
    processor=processor,
    out_batch_size= batch_size,
    device= 'cuda',
    precompute=False,
    load_activations_path=os.path.join(config.ACTIVATIONS,"medclip-rn50_chexpert_train.pt"),
)


val_activation_buffer = CLIPActivationBuffer.get(
    model_name,
    dataloader=dataloader['val'],
    modality='img',
    model=model,
    processor=processor,
    out_batch_size= batch_size,
    device= 'cuda',
    precompute=False,
    load_activations_path=os.path.join(config.ACTIVATIONS,"medclip-rn50_chexpert_val.pt")
)


#Trainer config
print(f"Setting up trainer config...")
base_config = {
    "activation_dim": activation_dim,
    "steps": steps,
    "warmup_steps": warmup_steps,
    "decay_start": decay_start,
    "device": device,
    "layer": layer,
    "lm_name": model_name,
    "submodule_name": submodule_name,
}

trainer_config = asdict(StandardTrainerConfig(
    **base_config,
    trainer=StandardTrainer,
    dict_class=AutoEncoder,
    sparsity_warmup_steps=sparsity_warmup_steps,
    lr=learning_rate,
    dict_size=dict_size,
    seed=seed,
    l1_penalty=l1_penalty,
    resample_steps=resample_steps,
    wandb_name=f"lr_{learning_rate}_l1_{l1_penalty}_rs_{resample_frq}_sparsitywu_{sparsity_warmup_frq}_decay{decay_start}",
))


#Training
print(f"Training...")
trainSAE(
    data=train_activation_buffer,
    trainer_configs=[trainer_config],
    use_wandb=use_wandb,
    epochs=n_epochs,
    save_steps=save_steps,
    save_dir=save_dir,
    log_steps=log_steps,
    eval_steps=eval_steps,
    eval_data=val_activation_buffer,
    wandb_project=wandb_project,
    wandb_entity=wandb_entity,
    normalize_activations=False,
    verbose=False,
    autocast_dtype=torch.bfloat16,
)



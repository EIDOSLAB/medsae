from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.buffer import CLIPActivationBuffer
from dictionary_learning.trainer_config import StandardTrainerConfig
from dictionary_learning.training import trainSAE
import config
import sweep_config
import random
import data_preprocess
import clip

import torchvision
import torch
import utils

from dataclasses import asdict


import os
#CONFIG
batch_size = 1024
activation_dim = 512
n_epochs = 200
device = 'cuda'
layer = 'Last'
model_name = "MedCLIP-RN50"
use_wandb = True
wandb_project = "SAE_MedCLIP_Chexpert_BatchTopK_sweep"
wandb_entity = ""
save_epochs = [2, 5, 10, 20, 50, 100, 150] 
log_steps = 2
eval_steps = 10
save_dir = config.SAE_CKPTS+f"/sweep/{model_name}/batchtopk/"

random.seed(sweep_config.random_seeds[0])
torch.manual_seed(sweep_config.random_seeds[0])


#Dataset preprocess
dataloader = utils.get_dataset('chexpert', batch_size)

steps = int(len(dataloader['train'])*n_epochs)





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



#list_architectures = ["standard_new","batch_top_k", "matryoshka_batch_top_k"]
list_architectures = ["batch_top_k"]
trainer_configs = sweep_config.get_trainer_configs(
    architectures= list_architectures,
    learning_rates= sweep_config.LEARNING_RATES,
    seeds= sweep_config.random_seeds,
    activation_dim=activation_dim,
    dict_sizes= sweep_config.DICT_SIZES,
    model_name=model_name,
    device=device,
    layer=layer,
    steps=steps,
    epochs=n_epochs,
)


#Training
print(f"Training...")
trainSAE(
    data=train_activation_buffer,
    trainer_configs=trainer_configs,
    use_wandb=use_wandb,
    epochs=n_epochs,
    save_epochs=save_epochs,
    save_dir=save_dir,
    log_steps=log_steps,
    eval_steps=eval_steps,
    eval_data=val_activation_buffer,
    wandb_project=wandb_project,
    wandb_entity=wandb_entity,
    normalize_activations=False,
    verbose=False,
    autocast_dtype=torch.float32,
)



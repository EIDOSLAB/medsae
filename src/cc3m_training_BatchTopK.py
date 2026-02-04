

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.buffer import CLIPActivationBuffer
from dictionary_learning.trainer_config import TopKTrainerConfig
from dictionary_learning.training import trainSAE
import config

import data_preprocess
import clip

import torchvision
import torch

from dataclasses import asdict


import os
#CONFIG
batch_size = 4096
expansion_factor = 8
activation_dim = 1024
k=100
n_epochs = 200
device = 'cuda'
layer = ''
model_name = 'CLIP-RN50'
submodule_name = ''
learning_rate = 5e-4
dict_size = expansion_factor*activation_dim
seed = None
warmup_rate = 0.0
decay_start_rate = 0.80
use_wandb = True
wandb_project = "SAE_CLIP-RN50_CC3M"
wandb_entity = ""
wandb_name = f"BatchTopK_k{k}_lr{learning_rate}_bs{batch_size}_exp{expansion_factor}_n_epochs{n_epochs}_wf{warmup_rate}"
save_dir = config.SAE_CKPTS+f"/{model_name}/{wandb_name}/"
log_steps = 2
eval_steps = 10


def get_used_memory():
    free, total = torch.cuda.mem_get_info(device)
    mem_used_MB = (total - free) / 1024 ** 2
    return mem_used_MB

#Dataset preprocess
dataset_path = '/scratch/medical-mi/Discover-then-Name/data/CC3M_TAR'
dataloader = data_preprocess.import_cc3m(dataset_path, batch_size)

n_batches = len(dataloader['train'])

steps = n_batches*n_epochs
save_steps = [ steps/i for i in range(1, n_epochs+1)]

warmup_steps = 0
decay_start = None

print(f"[Begin] Memory used : {get_used_memory()} MB")


#Model import
print(f"Importing model {model_name}...")
model, processor = clip.load("RN50")

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
    load_activations_path=os.path.join(config.ACTIVATIONS,"cc3m_train_clipRN50.pt"),
    shuffle=True,
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
    load_activations_path=os.path.join(config.ACTIVATIONS,"cc3m_val_clipRN50.pt"),
    shuffle=False,
)
print(f"[ActivationBuffer] Memory used : {get_used_memory()} MB")


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

trainer_config = asdict(TopKTrainerConfig(
    **base_config,
    trainer=BatchTopKTrainer,
    dict_class=BatchTopKSAE,
    lr=learning_rate,
    dict_size=dict_size,
    seed=seed,
    k=k,
    wandb_name=wandb_name,
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
    autocast_dtype=torch.float32,
)



from dataclasses import dataclass, asdict, field
from pickle import DICT
from typing import Optional, Type, Any
from enum import Enum

from torch import threshold
import torch as t
import math
import itertools

from dictionary_learning.trainers.standard import (
    StandardTrainer,
    StandardTrainerAprilUpdate,
)
from dictionary_learning.trainers.top_k import (
    TopKTrainer,
    AutoEncoderTopK,
)
from dictionary_learning.trainers.batch_top_k import (
    BatchTopKTrainer,
    BatchTopKSAE,
)
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKTrainer,
    MatryoshkaBatchTopKSAE,
)
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"
    Matryoshka_BATCH_TOP_K = "matryoshka_batch_top_k"

@dataclass
class SparsityPenalties:
    standard: list[float]
    standard_new: list[float]
    p_anneal: list[float]
    gated: list[float]


eval_num_inputs = 200
random_seeds = [0]

WARMUP_STEPS = 0
SPARSITY_WARMUP_STEPS = 0
DECAY_START_FRACTION = 0
#DICT_SIZES = [4096]
DICT_SIZES = [2048, 4096, 8192]
RESAMPLE_FRACTIONS = [0]
#RESAMPLE_FRACTIONS = [0, 0.05]
random_seeds = [0]

LEARNING_RATES = [1e-6,5e-5,1e-5,5e-4, 1e-4, 1e-3, 5e-3, 5e-2]
#LEARNING_RATES = [1e-5, 5e-4, 1e-4]

wandb_project = "MedCLIP_SAE_sweep"

SPARSITY_PENALTIES = SparsityPenalties(
    standard=[1e-4, 3e-3, 1e-3, 3e-2],
    standard_new=[1e-4, 3e-4, 7e-4, 1e-3, 3e-3, 3e-2],
    p_anneal=[0.006, 0.008, 0.01, 0.015, 0.02, 0.025],
    gated=[0.012, 0.018, 0.024, 0.04, 0.06, 0.08],
)

#TARGET_L0s = [100]
TARGET_L0s = [32, 64, 128, 256]
AUXK_ALPHA = [1/32]
THRESHOLD_START_STEP = [100000000]

@dataclass
class BaseTrainerConfig:
    activation_dim: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str
    warmup_steps: int
    steps: int
    decay_start: Optional[int]


@dataclass
class StandardTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]
    resample_steps: Optional[int] = None


@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    initial_sparsity_penalty: float
    sparsity_warmup_steps: Optional[int]
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: Optional[int] = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold


@dataclass
class MatryoshkaBatchTopKTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    k: int
    group_fractions: list[float] = field(
        default_factory=lambda: [
            (1 / 32),
            (1 / 16),
            (1 / 8),
            (1 / 4),
            ((1 / 2) + (1 / 32)),
        ]
    )
    group_weights: Optional[list[float]] = None
    auxk_alpha: float = 1 / 32
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000  # when to begin tracking the average threshold


@dataclass
class GatedTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    l1_penalty: float
    sparsity_warmup_steps: Optional[int]


@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    dict_size: int
    seed: int
    lr: float
    target_l0: int
    sparsity_warmup_steps: Optional[int]
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001


def get_trainer_configs(
    architectures: list[str],
    learning_rates: list[float],
    seeds: list[int],
    activation_dim: int,
    dict_sizes: list[int],
    model_name: str,
    device: str,
    layer: str,
    steps: int,
    epochs: int,
    warmup_steps: int = WARMUP_STEPS,
    sparsity_warmup_steps: int = SPARSITY_WARMUP_STEPS,
    decay_start_fraction=DECAY_START_FRACTION,
) -> list[dict]:
    decay_start = int(steps * decay_start_fraction) if decay_start_fraction != 0 else None

    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": layer,
    }
    if TrainerType.P_ANNEAL.value in architectures:
        for seed, dict_size, learning_rate, sparsity_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.p_anneal
        ):
            config = PAnnealTrainerConfig(
                **base_config,
                trainer=PAnnealTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                initial_sparsity_penalty=sparsity_penalty,
                wandb_name=f"PAnnealTrainer",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty, resample_frac in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard, RESAMPLE_FRACTIONS
        ):
            config = StandardTrainerConfig(
                **base_config,
                trainer=StandardTrainer,
                dict_class=AutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=16/(125*math.sqrt(dict_size)) if learning_rate is None else learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                resample_steps=int(steps / (resample_frac*epochs)) if resample_frac > 0 else None,
                wandb_name=f"StandardTrainer-size{dict_size}_lr{learning_rate}_l1{l1_penalty}_rs{resample_frac}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.STANDARD_NEW.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.standard_new
        ):
            config = StandardNewTrainerConfig(
                **base_config,
                trainer=StandardTrainerAprilUpdate,
                dict_class=AutoEncoderNew,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=16/(125*math.sqrt(dict_size)) if learning_rate is None else learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"StandardTrainerNew-size{dict_size}_lr{learning_rate}_l1{l1_penalty}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.GATED.value in architectures:
        for seed, dict_size, learning_rate, l1_penalty in itertools.product(
            seeds, dict_sizes, learning_rates, SPARSITY_PENALTIES.gated
        ):
            config = GatedTrainerConfig(
                **base_config,
                trainer=GatedSAETrainer,
                dict_class=GatedAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                l1_penalty=l1_penalty,
                wandb_name=f"GatedTrainer",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=TopKTrainer,
                dict_class=AutoEncoderTopK,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"TopKTrainer",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k, auxk_alpha, threshold_start_step in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s, AUXK_ALPHA, THRESHOLD_START_STEP
        ):
            config = TopKTrainerConfig(
                **base_config,
                trainer=BatchTopKTrainer,
                dict_class=BatchTopKSAE,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                auxk_alpha = auxk_alpha,
                threshold_start_step= threshold_start_step,
                k=k,
                wandb_name=f"BatchTopKTrainer-size{dict_size}_lr{learning_rate}_k{k}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.Matryoshka_BATCH_TOP_K.value in architectures:
        for seed, dict_size, learning_rate, k in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = MatryoshkaBatchTopKTrainerConfig(
                **base_config,
                trainer=MatryoshkaBatchTopKTrainer,
                dict_class=MatryoshkaBatchTopKSAE,
                lr=16/(125*math.sqrt(dict_size)) if learning_rate is None else learning_rate,
                dict_size=dict_size,
                seed=seed,
                k=k,
                wandb_name=f"MatryoshkaBatchTopKTrainer-size{dict_size}_lr{learning_rate}_k{k}",
            )
            trainer_configs.append(asdict(config))

    if TrainerType.JUMP_RELU.value in architectures:
        for seed, dict_size, learning_rate, target_l0 in itertools.product(
            seeds, dict_sizes, learning_rates, TARGET_L0s
        ):
            config = JumpReluTrainerConfig(
                **base_config,
                trainer=JumpReluTrainer,
                dict_class=JumpReluAutoEncoder,
                sparsity_warmup_steps=sparsity_warmup_steps,
                lr=learning_rate,
                dict_size=dict_size,
                seed=seed,
                target_l0=target_l0,
                wandb_name=f"JumpReluTrainer",
            )
            trainer_configs.append(asdict(config))

    return trainer_configs
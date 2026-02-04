# Mechanistic Interpretability for Medical Imaging Models with Sparse Autoencoders (SAEs)

This repository provides tools and scripts for **mechanistic interpretability** of medical imaging models using **Sparse Autoencoders (SAEs)**. It supports training, evaluation, and analysis of SAEs on neural network activations, with a focus on models such as **CLIP** and **MedCLIP**.  

![alt text](overview_illus.png "Overview of our proposed interpretability pipeline")

---

## 📑 Table of Contents

- [📂 Project Structure](#-project-structure)
- [⚙️ Installation](#-installation)
- [📊 Datasets](#-datasets)
- [🧠 Models](#-models)
- [💾 Computing Activations & SAE Features](#-computing-activations--sae-features)
- [🏋️ SAE Training](#-sae-training)
- [🗂️ Data Naming Convention](#-data-naming-convention)
- [🚀 Quick Start: Training an SAE, Assigning Concepts, and Evaluating Interpretability](#-quick-start-training-an-sae-assigning-concepts-and-evaluating-interpretability)
- [🔧 Useful Commands](#-useful-commands)
- [📚 References](#-references)

## 📂 Project Structure

```
.

├── data/                  # Data produced by the different scripts and steps of the scripts
│   ├── activations/       # Tensors of activations (advised naming : "model_dataset_split_additionalprocessing.pt")
│   ├── assign_concepts/   # CSV Tables of concept/neuron matchings
│   ├── probe_checkpoints/ # Tensors of probe weights trained with the script "train_probe.py"
│   ├── sae_checkpoints/   # Tensors of SAE weights (advised naming : "model_dataset_sae-architecture_remarks.pt")
│   ├── sae_features/      # Tensors of SAE features (or activations) (advised naming : "model_dataset_sae-architecture_remarks_sae-dataset_sae-split.pt")
│   └── vocabs/            # Txt vocabulary database
├── datasets/              # Datasets for training/evaluation (cc3m, chexpert, etc.)
├── dictionary_learning/   # Core dictionary learning package (SAE training, buffer, trainers) cloned from https://github.com/saprmarks/dictionary_learning/ and adapted for CLIP and MedCLIP models (see Section on Dictionnary_learning adaptation)
├── logs/                  # Training and evaluation logs
├── results/               # Output results (plots, metrics, etc.)
├── src/                   # Main source code for analysis, training, and plotting
│   ├── 1d_probe_training.py                # Train probes on 1D features for interpretability
│   ├── assign_names.py                     # Assign human-readable names from a word vocabulary to SAE features
│   ├── assign_names_VLM.py                 # Assign names using the vision-language model MedGEMMA
│   ├── cc3m_training.py                    # Train SAEs on the CC3M dataset
│   ├── cc3m_training_BatchTopK.py          # Train SAEs on CC3M with BatchTopK architecture
│   ├── chexpert_score.py                   # Evaluate CheXpert score (neuron-class correlation scores basically) for models/SAEs 
│   ├── chexpert_training.py                # Train SAEs on the CheXpert dataset
│   ├── compute_sweep_activations.py        # Compute activations for sweep experiments
│   ├── compute_sweep_deadlatents.py        # Analyze dead latents in sweep experiments
│   ├── config.py                           # Configuration for models, datasets, and paths
│   ├── data_preprocess.py                  # Data preprocessing utilities
│   ├── generate_medical_vocabulary.py      # Generate medical vocabulary files from a given text corpus
│   ├── linearity_sanity_check.py           # Sanity check for linearity in features by analyzing combined images (see Report)
│   ├── linearity_sanity_check_pairs.py     # Linearity check for combined pairs
│   ├── linearity_sanity_check_negative_allowed.py # Linearity check allowing negative least square solving
│   ├── monosemanticity_score.py            # Compute monosemanticity score for SAEs/Models activations (see https://arxiv.org/pdf/2504.02821v1)
│   ├── monosemanticity_score_actversion.py # Alternative monosemanticity score computation (adaptation of https://arxiv.org/abs/2501.18052)
│   ├── naming_eval.py                      # Evaluate naming quality for SAE features with MedGemma (see Report)
│   ├── neuron_class_correlation_comparaison.py # Compare neuron-class correlations 
│   ├── plot_feature_sparsity_distribution.py    # Plot feature sparsity distributions
│   ├── plot_naming_eval.py                 # Plot results of naming evaluation (score distribution)
│   ├── plot_neuron_class_correlation.py    # Plot neuron-class correlation results (correlation matrix)
│   ├── plot_top_activating_images.py       # Visualize top activating images for features
│   ├── save_activations_or_features.py     # Save activations or SAE features to disk
│   ├── sweep.py                            # Run parameter sweeps for training/evaluation
│   ├── sweep_config.py                     # Configuration for sweep experiments
│   ├── train_probe.py                      # Train probes on SAE or model features
│   └── utils.py                            # General utility functions such as data process, model import, etc..
├── vocab/                 # Vocabulary files
├── build.sh               # Build script
├── Dockerfile*            # Docker configurations
├── requirements.txt       # Python dependencies
├── README.md              # (This file)
└── LICENSE
```

---

## ⚙️ Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Install dictionary_learning:

```bash
pip install -e dictionary_learning/
```

Install transformers version 4.24.0:
```bash
pip install transformers==4.24.0
```

Fix current medclip library issue :

Add strict=False in this file:
https://github.com/RyanWangZf/MedCLIP/blob/main/medclip/modeling_medclip.py#L185
(See : https://github.com/RyanWangZf/MedCLIP/issues/37)

---

## 📊 Datasets

Supported datasets:  
`["chexpert", "cc3m", "mimic-cxr", "cifar100", "cifar10", "places365"]`

- **Torch datasets** (`cifar100`, `cifar10`, `places365`) → automatically downloaded.  
- **External datasets** require manual setup:  
  - **CC3M**: `datasets/cc3m/` with `cc3m-train-{0000..0575}.tar` and `cc3m-validation-{0000..0015}.tar`  
  - **CheXpert**: `datasets/chexpert/` with CSVs (`chexpert_train.csv`, `chexpert_valid.csv`, `chexpert_train_shuffled.csv` optional). Must include column `Path` (image paths) + one column per label (0/1).  
  - **MIMIC-CXR**: downloaded via Hugging Face (authentication may be required).  

**Usage:** Select dataset via `--dataset` tag in scripts. Some scripts are dataset-specific (e.g., `chexpert_score.py`).

The list of supported datasets can be easily expended by modifying "utils.py" and "data_preprocess.py".

---

## 🧠 Models

Supported models:  
`["MedCLIP-RN50", "CLIP-RN50", "MedCLIP-ViT"]`

- **CLIP models**: from [OpenAI CLIP](https://github.com/openai/CLIP)  
- **MedCLIP models**: from [MedCLIP repo](https://github.com/RyanWangZf/MedCLIP)  

Extend by editing `src/utils.py` + `src/config.py`, and adding buffers in `dictionary_learning/buffer.py`.

---

## 💾 Computing Activations & SAE Features

Activations and SAE features can be computed either:  
- **Independently** using `src/save_activations_or_features.py`  
- **On the fly** within other scripts that require them  

Because these computations are time-consuming, the workflow is designed so that they are performed **once**, saved in the `data/` directory (under `data/activations/` or `data/sae_features/`), and then **reused automatically**.  

Each script checks whether the requested activations or features already exist (see [Data Naming Convention](#️-data-naming-convention)):  
- ✅ If available → they are loaded from disk  
- ❌ If not → they are computed and saved for future use  

You can also provide **custom precomputed data** with:  
- `--load_activations_path` → specify a path to activations  
- `--load_features_path` → specify a path to SAE features  

This enables you to preprocess data manually and still use it seamlessly in the scripts.  

### 🔄 Buffered Loading
The repository implements **buffered loading** via `FeatureBuffer` and `CLIPActivationBuffer` (customized from the `dictionary_learning` library).  
- This avoids loading entire datasets into memory, reducing **RAM usage**.  
- Some scripts, however, bypass this and load all data directly into memory by iterating over the buffer.  

---

## 🏋️ SAE Training

SAEs are trained with **our customized version** of `dictionary_learning/training.py`.  

Example scripts:  
- `src/cc3m_training.py`  
- `src/chexpert_training.py`  
- `src/sweep.py`, `src/sweep_config.py`  

Available SAE architectures:  
`["Standard", "BatchTopKSAE", "StandardAprilUpdate", "MatryoshkaBatchTopKSAE"]`  

Details: [dictionary_learning repo](https://github.com/saprmarks/dictionary_learning/).

---

## 🗂️ Data Naming Convention

- **SAE features**: `model_dataset_sae-arch_remarks_sae-dataset_sae-split.pt`  
- **SAE checkpoints**: `model_dataset_sae-arch_remarks.pt`  
- **Activations**: `model_dataset_split_additionalprocessing.pt`

---

## 🚀 Quick Start: Training an SAE, Assigning Concepts, and Evaluating Interpretability

Before starting, make sure you have completed the [installation](#️-installation) steps.  
This guide walks through the process step by step, using intermediate outputs so that errors can be isolated easily.

---

### 1. Choose a Model
Example:  
```bash
MedCLIP-RN50
```

---

### 2. Choose and Prepare a Dataset
- If using a Torch dataset (e.g., `cifar100`, `cifar10`, `places365`), it will download automatically.  
- If using **CheXpert**, download the dataset (e.g., [Kaggle CheXpert](https://www.kaggle.com/datasets/ashery/chexpert/data)) and check that paths in the CSV files point to the correct image locations.

---

### 3. Compute Activations
From the project root, compute train split activations:
```bash
python src/save_activations_or_features.py   --model_name MedCLIP-RN50   --split train   --dataset chexpert
```
This saves activations to:
```
data/activations/MedCLIP-RN50_chexpert_train.pt
```

For the validation split:
```bash
python src/save_activations_or_features.py   --model_name MedCLIP-RN50   --split val   --dataset chexpert
```

---

### 4. Train an SAE
Use the training scripts provided (e.g., `src/chexpert_training.py` or `src/cc3m_training.py`).  

⚠️ **Important:**  
- By default, `chexpert_training.py` uses **CheXpert + MedCLIP-RN50** with the **Standard SAE** architecture.  
- Modify training parameters (e.g., `batch_size`, `learning_rate`, `SAE architecture`) directly in the script.  
- To change SAE type, update the trainer imports:  
  ```python
  from dictionary_learning.trainers.standard import StandardTrainer
  from dictionary_learning.trainer_config import StandardTrainerConfig
  from dictionary_learning.dictionary import AutoEncoder
  ```
- Refer to the `dictionary_learning` library and sweep scripts for customization examples.

Run training:
```bash
python src/chexpert_training.py
```

Checkpoints will be saved in:
```
data/sae_checkpoints/
```

---

### 5. Compute SAE Features
Once training is done, compute SAE features with:
```bash
python src/save_activations_or_features.py   --model_name MedCLIP-RN50   --split val   --dataset chexpert   --sae_path "MedCLIP-RN50_chexpert_standard.pt"   --sae_type_name "Standard"
```

Output is saved in:
```
data/sae_features/MedCLIP-RN50_chexpert_standard_chexpert_val.pt
```

👉 You can also run this for other datasets/splits. If the corresponding activations are missing, both activations and SAE features will be computed automatically.

---

### 6. Assign Concepts to SAE Features
Two methods are supported:
1. **Using MedGEMMA VLM** → analyzes top/low activating images (`src/assign_names_VLM.py`)  
2. **Using a vocabulary** → matches words/sentences via cosine similarity in CLIP space (`src/assign_names.py`)

Example with **MedGEMMA**:
```bash
python src/assign_names_VLM.py   --sae_path MedCLIP-RN50_chexpert_standard.pt   --sae_type_name Standard   --model_name MedCLIP-RN50   --split val   --dataset chexpert
```

This saves a CSV in:
```
data/assign_concepts/MedCLIP-RN50_chexpert_standard_reportsFalse.csv
```

The file includes:
- `neuron` → feature index  
- `concept` → assigned name  
- `description` → explanation  

⚠️ You’ll need to configure Hugging Face CLI for MedGemma:  
[MedGemma model page](https://huggingface.co/google/medgemma-4b-it)

---

### 7. Evaluate Neuron Naming Interpretability
Evaluation uses MedGemma as a classifier:  
- It predicts whether an image activates a neuron based on the concept description.  
- Results are compared against ground truth activations.  

Run:
```bash
python src/assign_names_VLM.py   --sae_path MedCLIP-RN50_chexpert_standard.pt   --sae_type_name Standard   --model_name MedCLIP-RN50   --split val   --dataset chexpert   --assigned_concepts MedCLIP-RN50_chexpert_standard_reportsFalse.csv
```



---

## 🔧 Useful Commands

- **Plot feature sparsity distribution**  
```bash
python src/plot_feature_sparsity_distribution.py   --sae_path clip-rn50_cc3m_repo.pt   --sae_type_name Standard   --model_name CLIP-RN50   --dataset cc3m   --split val
```

- **Visualize top activating images**  
```bash
python src/plot_top_activating_images.py   --sae_path clip-rn50_cc3m_repo.pt   --sae_type_name Standard   --model_name CLIP-RN50   --dataset places365   --split val
```

- **Compute CheXpert score**  
```bash
python src/chexpert_score.py   --model_name MedCLIP-RN50   --split trainval   --dataset chexpert   --sae_path medclip-rn50_chexpert_standard.pt   --sae_type_name Standard
```

---

## 📚 References

- [CLIP (OpenAI)](https://github.com/openai/CLIP)  
- [MedCLIP](https://github.com/RyanWangZf/MedCLIP)  
- [Dictionary Learning](https://github.com/saprmarks/dictionary_learning)  

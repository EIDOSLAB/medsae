




# Paths
SAE_CKPTS = "./data/sae_checkpoints"
CONCEPTS = "./data/assign_concepts"
ACTIVATIONS = "./data/activations"
FEATURES =  "./data/sae_features"
RESULTS = "./results"
DATASETS = "./datasets"
VOCABS = "./data/vocabs"

CHEXPERT_LOW_LABELS = ["No Finding", "Support Devices", "Fracture", "Edema", "Consolidation", "Pneunomia", "Lung Lesion", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Cardiomegaly"]
CHEXPERT_HIGH_LABELS = ["No Finding", "Support Devices", "Fracture", "Enlarged Cardiomediastinum", "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumothorax"]
CHEXPERT_LABELS = ['No Finding','Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
            'Fracture', 'Support Devices']

AVAILABLE_MODELS = ["MedCLIP-RN50", "CLIP-RN50", "MedCLIP-ViT"]
TORCH_DATASETS = ["cifar100", "cifar10", 'places365']
AVAILABLE_DATASETS = TORCH_DATASETS+ ["chexpert", "cc3m", "mimic-cxr"]
AVAILABLE_SAE = ["Standard", "BatchTopKSAE", "StandardAprilUpdate", "MatryoshkaBatchTopKSAE"]
SAE_TRAINER_TO_SAE = {
    "BatchTopKTrainer" : "BatchTopKSAE",
    "StandardTrainer" : "Standard",
    "StandardTrainerNew" : "StandardAprilUpdate",
    "MatryoshkaBatchTopKTrainer" : "MatryoshkaBatchTopKSAE"
}

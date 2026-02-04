import clip
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE, MatryoshkaBatchTopKTrainer
from dictionary_learning.trainers.standard import StandardTrainer, StandardTrainerAprilUpdate
from dictionary_learning.dictionary import AutoEncoder, AutoEncoderNew
import data_preprocess
import config
import functools
import os


def get_model(model_name):
    if model_name not in config.AVAILABLE_MODELS:
        raise Exception(f"ERROR : {model_name} is not available for this script, the model name should be in {config.AVAILABLE_MODELS}")

    if model_name == "MedCLIP-ViT":
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
        processor = MedCLIPProcessor()
    elif model_name == "MedCLIP-RN50":
        model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        model.from_pretrained()
        processor = MedCLIPProcessor()
        # processor.feature_extractor.do_convert_rgb = False
        # processor.feature_extractor.do_pad_square = False
    elif model_name.startswith("CLIP"):
        model, processor = clip.load(model_name[5:])
    
    return model, processor


def get_dataset(name, batch_size, labels=False, shuffled_version=False):
    if name not in config.AVAILABLE_DATASETS:
        raise Exception(f"ERROR : {name} is not available for this script") 
    if name in config.TORCH_DATASETS:
        return data_preprocess.import_torch_dataset(name, batch_size, labels)
    elif name == "cc3m":
        return data_preprocess.import_cc3m(os.path.join(config.DATASETS,name), batch_size)
    elif name == 'chexpert':
        return data_preprocess.import_chexpert(os.path.join(config.DATASETS,name), batch_size, labels, shuffled_version)
    elif name == 'mimic-cxr':
        return data_preprocess.import_mimic_cxr(os.path.join(config.DATASETS,name), batch_size, labels)



def activations_already_precomputed(model_name, dataset_name, split, layer=None):
    if model_name not in config.AVAILABLE_MODELS:
        raise Exception(f"ERROR : {model_name} is not available for this script")
    if dataset_name not in config.AVAILABLE_DATASETS:
        raise Exception(f"ERROR : {dataset_name} is not available for this script") 

    if layer is not None:
        path = os.path.join(config.ACTIVATIONS, f"{model_name.lower()}_{layer.lower()}_{dataset_name.lower()}_{split.lower()}.pt")
    else:
        path = os.path.join(config.ACTIVATIONS, f"{model_name.lower()}_{dataset_name.lower()}_{split.lower()}.pt")
    if not os.path.exists(path):
        return (False,path)
    return (True, path)

def features_already_precomputed(sae_name, dataset_name, split, layer=None):
    if dataset_name not in config.AVAILABLE_DATASETS:
        raise Exception(f"ERROR : {dataset_name} is not available for this script") 

    if layer is not None:
        path = os.path.join(config.FEATURES, f"{sae_name.lower()}_{layer.lower()}_{dataset_name.lower()}_{split.lower()}.pt")
    else:
        path = os.path.join(config.FEATURES, f"{sae_name.lower()}_{dataset_name.lower()}_{split.lower()}.pt")
    if not os.path.exists(path):
        return (False,path)
    return (True, path)

    
def get_sae_architecture(sae_type_name):
    if sae_type_name == "BatchTopKSAE":
        return (BatchTopKSAE, BatchTopKTrainer)
    elif sae_type_name == "Standard":
        return (AutoEncoder, StandardTrainer)
    elif sae_type_name == "StandardAprilUpdate":
        return (AutoEncoderNew, StandardTrainerAprilUpdate)
    elif sae_type_name == "MatryoshkaBatchTopKSAE":
        return (MatryoshkaBatchTopKSAE, MatryoshkaBatchTopKTrainer)

    else:
        raise Exception(f"ERROR : {sae_type_name} is not available for this script")

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def get_common_parser():
    pass
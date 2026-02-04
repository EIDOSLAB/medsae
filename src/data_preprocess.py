import torchvision
import torch
import os
import webdataset as wds
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
import pandas as pd
import config
from webdataset import WebLoader


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_paths = pd.read_csv(csv_file)['Path'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
    

class ChexpertLabelDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with image paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_name = ['No Finding','Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
            'Fracture', 'Support Devices']
        
        df = pd.read_csv(csv_file)[self.labels_name]
        self.labels = torch.tensor(df.values, dtype=torch.float32)
        self.shape = self.labels.shape

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        labels_ = self.labels[idx]
        return labels_

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels=False, transform=None):
        self.transform = transform
        self.labels = labels
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.labels:
            return self.dataset[idx][1]
        else:
            img = self.dataset[idx][0]
            if self.transform:
                img = self.transform(img)
            return img
    

def import_torch_dataset(name, batch_size, labels=False):
    dataset_path = os.path.join(config.DATASETS, name)
    if name == "cifar10":
        train = torchvision.datasets.CIFAR10(
                    root=dataset_path, train=True, download=True)
        val = torchvision.datasets.CIFAR10(
                    root=dataset_path, train=False, download=True)
    elif name == "cifar100":
        train = torchvision.datasets.CIFAR100(
                    root=dataset_path, train=True, download=True)
        val = torchvision.datasets.CIFAR100(
                    root=dataset_path, train=False, download=True)
    elif name == "places365":
        train = torchvision.datasets.Places365(
                    root=dataset_path, split='train-standard', small=True, download=True)
        val = torchvision.datasets.Places365(
                    root=dataset_path, split='val', small=True, download=True)
    loader = { 'train' : DataLoader(TorchDataset(train), batch_size=batch_size, shuffle=False, collate_fn=lambda x: x),
             'val' : DataLoader(TorchDataset(val), batch_size=batch_size, shuffle=False, collate_fn=lambda x: x) }
    if labels:
        return loader, { 'train' : DataLoader(TorchDataset(train, labels=True), batch_size=batch_size, shuffle=False),
             'val' : DataLoader(TorchDataset(val, labels=True), batch_size=batch_size, shuffle=False) }
    else:
        return loader


class CustomDataCollatorImg:
    def __init__(self) -> None:
        pass

    def __call__(self, batch):
        imgs = [i["image"] for i in batch]
        #imgs = torch.stack(imgs)
        return imgs

class TorchHFImageDataset(Dataset):
    def __init__(self, hf_ds, transform=None):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if self.transform:
            img = self.transform(img)
        return img

def import_mimic_cxr(save_dir, batch_size, labels=False):
    """
    Load the itsanmolgupta/mimic-cxr-dataset from Hugging Face.
    Returns train DataLoader and an empty val DataLoader.
    If labels=True, also returns separate label loaders.
    """
    # Load only train split
    dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split="train", cache_dir=save_dir)
    # Define image preprocessing
    transform = T.Compose([
        T.ToTensor(),
        T.ToPILImage()
    ])
    train_ds = TorchHFImageDataset(dataset, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    # Create dummy empty val loader
    val_loader = DataLoader(TorchHFImageDataset(dataset.select([]), transform=transform),
                            batch_size=batch_size)

    loaders = {'train': train_loader, 'val': val_loader}

    return loaders

def import_cc3m(dataset_path, batch_size):
    train_shard     = os.path.join(dataset_path, "cc3m-train-{0000..0575}.tar")
    val_shard       = os.path.join(dataset_path, 'cc3m-validation-{0000..0015}.tar')
    shard_list = [train_shard, val_shard]
    dataloader = {'train' : None,'val' : None}
    type_list = ['train', 'val']
    for split, shard in zip(type_list, shard_list):
        pipeline = [
            wds.SimpleShardList(shard),
            # at this point we have an iterator over all the shards
        ]
        transform = torchvision.transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Resize((224,224)),
                # torchvision.transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            ])
        pipeline.extend(
            [
                wds.split_by_worker,
                # at this point, we have an iterator over the shards assigned to each worker
                wds.tarfile_to_samples(),
                # wds.select(filter_no_caption_or_no_image),
                wds.decode("pilrgb"),
                wds.rename(image="jpg;png;jpeg"),
                #wds.map_dict(image=transform),
        ])

        collator = CustomDataCollatorImg()
        pipeline.extend(
            [wds.batched(batch_size, partial=False, collation_fn=collator)])
        
        if split == 'train': length = 2905954
        else: length = 13443
        data = wds.DataPipeline(*pipeline).with_length(int(length/batch_size))
        dataloader[split] = DataLoader(data, batch_size=None, shuffle=False, num_workers=0)
    return dataloader

class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")

def import_chexpert(dataset_path, batch_size, labels=False, shuffled_version=False, balanced_val_target=200):
    transform = torchvision.transforms.Compose([
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.ColorJitter(0.1,0.1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((224,224)),
                #torchvision.transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304]),
                torchvision.transforms.ToPILImage()
            ])
    if shuffled_version:
        train = ImageDataset(dataset_path+"/chexpert_train_shuffled.csv", transform=transform)
        train_labels = ChexpertLabelDataset(dataset_path+"/chexpert_train_shuffled.csv")
    else:
        train = ImageDataset(dataset_path+"/chexpert_train.csv", transform=transform)
        train_labels = ChexpertLabelDataset(dataset_path+"/chexpert_train.csv")
    val = ImageDataset(dataset_path+"/chexpert_valid.csv", transform=transform)

    
    val_labels = ChexpertLabelDataset(dataset_path+"/chexpert_valid.csv")

    trainval, trainval_labels = get_chexpert5x200(train, train_labels, balanced_val_target)

    dataset = {'train' : train, 'val' : val, 'trainval' : trainval}
    loader = {'train' : None, 'val' : None, 'trainval' : None}
    labels_loader = {'train' : torch.utils.data.DataLoader(train_labels,batch_size=batch_size, shuffle=False), 'val' : torch.utils.data.DataLoader(val_labels,batch_size=batch_size, shuffle=False), 'trainval' : torch.utils.data.DataLoader(trainval_labels,batch_size=batch_size, shuffle=False)}
    
        
    type_list = ['train', 'val', 'trainval']
    for split in type_list:
        loader[split] = torch.utils.data.DataLoader(dataset[split],batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    if labels:
        return loader, labels_loader
    else: return loader
        
def get_chexpert5x200(train_dataset, train_labels, target_per_class=200):
    labels = train_labels.labels  # shape: (N, C), binary multi-label matrix
    num_classes = labels.shape[1]
    target_per_class = target_per_class

    class_counts = torch.zeros(num_classes, dtype=torch.long)
    selected_indices = []

    # Shuffle indices for random selection
    all_indices = torch.arange(len(labels))

    for idx in all_indices:
        image_labels = labels[idx]  # shape: (C,)
        contributing_classes = torch.where(image_labels == 1)[0]

        # Skip images with no labels
        if len(contributing_classes) == 0: continue

        # Skip image if any class would go over the target
        if any(class_counts[c] >= target_per_class for c in contributing_classes):
            continue

        # Otherwise, accept the image
        selected_indices.append(idx.item())
        for c in contributing_classes:
            class_counts[c] += 1

        # Stop early if all classes reached 200
        if torch.all(class_counts >= target_per_class):
            break


    # Create final dataset
    selected_indices = sorted(selected_indices)
    final_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
    final_labels = torch.utils.data.Subset(train_labels, selected_indices)

    return final_dataset, final_labels


def import_vocab(vocab_dataset, batch_size):
    with open(vocab_dataset, 'r') as file:
        lines = file.readlines()
    lines = [x.strip().replace("\n",'') for x in lines]
    loader = torch.utils.data.DataLoader(lines, batch_size=batch_size, shuffle=False)
    return loader
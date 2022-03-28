import os
import pdb
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ColorJitter, Normalize, RandomHorizontalFlip, RandomResizedCrop, RandomApply, Resize
import torchvision.transforms.functional as TF


def get_cooking_dataset_dataloaders(config):
    dataloaders = {}
    for split in ["train", "validation", "test"]:
        dataset = CookingStateDataset(split=split, transform=split == "train")
        dataloaders[split] = DataLoader(dataset, batch_size=config["bs"], num_workers=config["nw"], shuffle=split == "train")
    return dataloaders


class CookingStateDataset(Dataset):
    def __init__(self, base_dir="/data01/mohit/CookingDataset/cooking_dataset", split='train', transform=False, seed=0):
        assert split in ["train", "test", "validation"]
 
        self.split = split
        label_key = "label"

        vid_info = pd.read_csv(Path(base_dir) / "state_annotations.csv")
        self.classes = sorted(list(set(vid_info["label"].values)))
        self.num_classes = len(self.classes)

        vid_info = vid_info[vid_info['split'] == (split if split != 'validation' else 'valid')]
        vid_info = vid_info.sort_values("file")
        self.labels = [i for i in vid_info[label_key]]

        # define transforms
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        if transform:
            self.transform = Compose([
                RandomResizedCrop((128, 128), scale=(0.8, 1.0)),
                RandomApply(torch.nn.ModuleList([
                    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    ]), p=0.5),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                normalize
            ])
        else:
            self.transform = Compose([Resize((128, 128)), ToTensor(), normalize])

        # ensure well defined order
        self.labels = [i for i in vid_info[label_key]]
        self.track_files = [f for f in vid_info['file']]

        self.label_indexes = [self.classes.index(l) for l in self.labels]

        print(f"loaded {split} dataset: {len(self.track_files)} data points")

    def __len__(self):
        return len(self.track_files)

    def __getitem__(self, item):
        img = Image.open(self.track_files[item])
        img_small_dim = min(img.size)
        img = TF.center_crop(img, (img_small_dim, img_small_dim))
        img = self.transform(img)
        
        clss = self.label_indexes[item]
        return img, clss, 0
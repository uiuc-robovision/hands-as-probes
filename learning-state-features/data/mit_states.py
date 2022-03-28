import os
import pdb
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .transforms import get_mitstates_transform

from paths import *

valid_nouns = \
[      'pear', 'salmon', 'vegetable', 
       'kitchen', 'ribbon'
       'door', 'velvet', 'coat', 'wall', 'book', 'banana',
       'granite', 'toy', 'cheese', 'room', 'bread', 'leaf',
       'pie', 'box', 'soup', 
       'handle', 'chocolate', 'paste', 'ceramic', 
       'pizza', 'meat',
       'berry', 'thread', 'wire', 'glass', 
       'jewelry', 'cotton', 'butter', 'card', 'pot', 'cookie', 'envelope',
       'mud', 'lightbulb', 'glasses', 
       'shower', 'shorts', 'plant', 'fan',
       'tile', 'fish', 'clothes', 'wool', 'ice', 'bathroom',
       'paper',  'desk', 'carpet', 'foam', 'diamond', 'steel',
       'tomato', 'wheel', 'bean', 'fabric', 'beef',
       'belt', 'sauce', 'basement', 'brass', 'cabinet', 
       'bag', 'shirt', 'eggs', 'salad', 'mirror', 'steps', 
       'plastic', 'dress', 'blade', 'phone', 'knife',
       'rubber', 'wood', 'fruit', 
       'metal', 'clay', 'bracelet',
       'apple', 'garage', 'jacket', 'paint', 'potato', 'fig', 'frame',
       'bed', 'chair',
       'bowl', 'coffee',
       'rope', 'ball', 
       'oil', 'chicken', 'balloon', 'milk', 'candle', 'pasta', 'sugar',
       'cord', 'plate', 'pool', 'sandwich',
       'garlic', 'newspaper', 'screw', 'necklace',
       'penny', 'wax', 'keyboard', 'tube',
       'tea', 'pants', 'bottle', 'persimmon', 
       'hat', 'mat', 'lemon',
       'window', 'seafood', 'aluminum', 'coin',
       'computer', 'orange', 'gemstone', 'clock',
       'flower', 'camera', 'water', 'table',
       'laptop', 'floor', 'log', 'shoes', 'bucket', 'cable',
       'dust', 'cake', 'basket', 'tie',
       'dirt', 'nut', 'palm', 'candy',
       'key', 'drum'
]

def split_data(df, r=0.8):
    keys = ["train_imgs", "train_labels", "validation_imgs", "validation_labels"]
    out = {k: [] for k in keys}
    dirs = df["adj"] + " " + df["noun"]
    dirset = dirs.unique()
    for d in dirset:
        subset = df[dirs == d].sample(frac=1)
        dftrain, dfval = subset.iloc[:int(len(subset) * r)], subset.iloc[int(len(subset) * r):]
        out["train_imgs"] += list(dftrain.path)
        out["validation_imgs"] += list(dfval.path)
        out["train_labels"] += list(dftrain.adj_id)
        out["validation_labels"] += list(dfval.adj_id)
    return out

def get_mitstates_dataloaders(config, filtered_nouns=False, adj=True):
    df = pd.read_csv(f"{DIR_MIT_STATES}/release_dataset/meta_info/info.csv")

    classes = None
    dic_opp = None
    if adj:
        classes = df.adj.unique()
        with open(DIR_MIT_STATES / "release_dataset" / "adj_ants.csv") as f:
            lines = [l.strip().split(",")[:-1] for l in f.readlines()[1:]]
            dic_opp = {line[0]:line[1:] for line in lines}
            del lines
    elif filtered_nouns:
        classes = valid_nouns
        df = df[df.noun.isin(valid_nouns)]
    else:
        classes = df.noun.unique()
    assert classes is not None

    dataloaders = {}
    for split in ["train", "test"]:
        dataset = MITStates(
            DIR_MIT_STATES / "release_dataset", 
            df[df.split == split].path.values, 
            df[df.split == split].adj.values if adj else df[df.split == split].noun.values,
            classes, transform=split == "train", 
            dic_opp=dic_opp
        )
        dataloaders[split] = DataLoader(dataset, batch_size=config["bs"], num_workers=config["nw"], shuffle=split == "train")
    dataloaders["validation"] = dataloaders["test"]
    return dataloaders


class MITStates(Dataset):
    def __init__(self, base_dir, imgfiles, labels, classes, transform=False, dic_opp=None):
        self.imgfiles = [Path(base_dir) / "images" / img for img in imgfiles]
        self.labels = labels
        self.classes = classes 
        self.transform = get_mitstates_transform(transform, size=128)
        assert len(self.labels) == len(self.imgfiles)
     
        self.num_classes = len(self.classes)
        self.labels_to_clsindex = {classes[i]:i for i in range(self.num_classes)}

        self.dic_opp = dic_opp
        self.weights = torch.ones(self.labels.shape)
        if self.dic_opp is not None:
            self.class2num = {j: i for i, j in enumerate(self.classes)}
            self.num2pos2numneg = {self.class2num[i]: [self.class2num[k] for k in j if k != ''] for i, j in self.dic_opp.items()}
            self.vectorized_labels = F.one_hot(torch.tensor([self.labels_to_clsindex[label] for label in self.labels])).float()
            # pdb.set_trace()
            self.weights = self.vectorized_labels.clone()
            for i, j in self.num2pos2numneg.items():
                indices = self.weights[:, i] == 1
                for ind in j:
                    self.weights[indices, ind] = 1
        # pdb.set_trace()

    def __len__(self):
        return len(self.imgfiles)

    def __getitem__(self, item):
        img = Image.open(self.imgfiles[item]).convert('RGB')
        labels = self.labels_to_clsindex[self.labels[item]]
        if self.dic_opp is not None:
            labels = self.vectorized_labels[item]
        weights = self.weights[item]
        return self.transform(img), labels, weights
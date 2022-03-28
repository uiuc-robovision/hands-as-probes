import os
import shutil
import sys
import argparse
from pathlib import Path
from PIL import Image
from dominate.tags import track
from numpy.lib.arraysetops import isin
from torch.utils.data.dataset import Dataset
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb
from html4vision import Col, imagetable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import functional as F

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from pretraining.data.epic_kitchen_object_crops import EpicKitchenObjectCrops

class ObjectCrops(EpicKitchenObjectCrops):
    gun71_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    def __init__(self, base_dir, split='validation', tracks_dir='tracks'):
        super(ObjectCrops, self).__init__(base_dir, split, 1.0, tracks_dir)

    def __getitem__(self, item):
        path, idx = self.object_crops[item]
        image = Image.open(path).crop((128*idx, 0, (idx+1)*128, 128))
        return {"path": str(path), "idx": idx, "image": self.gun71_transform(image)}

class Visualization:
    def __init__(self, args):
        self.device = torch.device(args.gpu)
        

    def get_embeddings(self):
        splits = [s +"_ioumf" for s in ["validation"]]
        datasets = {split:ObjectCrops(self.data_dir, split) for split in splits}
        data = {
            key: DataLoader(val, batch_size=4096, num_workers=16, pin_memory=True) 
            for key, val in datasets.items()
        }

        predictions = {"path": [], "idx":[], "embedding":[]}
        for split, dl in data.items():
            for dic in tqdm(dl, desc=split, dynamic_ncols=True):
                embeddings = self.model(dic["image"])
                predictions["path"] += dic["path"]
                predictions["idx"] += dic["idx"]
                predictions["embedding"].append(embeddings)
        
        predictions["path"] = np.array(predictions["path"])
        predictions["idx"] = np.array(predictions["idx"])
        predictions["embedding"] = torch.cat(predictions["embedding"])
        return predictions
  



if __name__ == '__main__':
    pass
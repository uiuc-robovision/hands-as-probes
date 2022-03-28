import os
import pdb
import glob
import logging
import numpy as np
import pandas as pd
from PIL import Image
import scipy.misc
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, ColorJitter, Normalize, RandomHorizontalFlip, RandomResizedCrop, RandomApply
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import Resize


def select_k(df, data_percent, classes, column, dic_opp, label_key, opposite=False, seed=0):
    pos_list = []
    neg_list = []
    df = df.copy()
    for klass in classes:
        labels = [lab.split(",") for lab in df[column]]
        outp = df[[klass in i for i in labels]].copy()
        if opposite:
            inds = [np.array([j in i for i in labels]) for j in dic_opp[klass]]
            ind = inds[0] if len(inds) == 1 else np.logical_or.reduce(inds)
            outn = df[ind].copy()
        else:
            outn = df[[klass not in i for i in labels]].copy()
        
        outp[label_key] = klass
        outn[label_key] = klass
        outp['flag'] = 1
        outn['flag'] = -1
        psampled = outp.sample(frac=data_percent/100, random_state=seed)
        nsampled = outn.sample(frac=data_percent/100, random_state=seed)

        pos_list.append(psampled)
        neg_list.append(nsampled)

    pos_frame = pd.concat(pos_list, axis=0)
    neg_frame = pd.concat(neg_list, axis=0)
    filtered_vid_info = pd.concat([pos_frame, neg_frame], axis=0)
    return filtered_vid_info


class StateDataLoader(Dataset):

    def __init__(self, logger, config, data_dir, split='train', transform=False, dic_opp=None, sample_opposite=False, percent=None, seed=0):
        label_key = "state"
        path_key = "file"
        self.data_dir = data_dir
        self.filepath_map = {Path(p).name:p for p in glob.glob(f"{self.data_dir}/**/*.jpg", recursive=True)}

        self.split = split
        vid_info = pd.read_csv("./data/epic-states-annotations.csv")
        vid_info = vid_info[vid_info['split'] == split]

        # filter files according to criteria
        filter = config["data"].get("filter", None)
        train_filter = config["data"].get("filter_train", None)
        eval_filter = config["data"].get("filter_eval", None)
        if filter is not None:
            filtered_ids = [vid_info[key].isin(val).values for key, val in filter.items()]
            filtered_ids = filtered_ids[0] if len(filtered_ids) == 1 else np.logical_and(*filtered_ids)
            vid_info = vid_info[filtered_ids]

        # Novel classes setting
        if split == 'train' and train_filter is not None:
            logger.info(f"Filtering train: {train_filter}")
            vid_info = vid_info[[vid_info[key].isin(val) for key, val in train_filter.items()][0]]
        elif split != 'train' and eval_filter is not None:
            logger.info(f"Filtering {split}: {eval_filter}")
            vid_info = vid_info[[vid_info[key].isin(val) for key, val in eval_filter.items()][0]]

        filtered_ids = vid_info[label_key].str.contains('|'.join(sorted(list(dic_opp.keys()))))
        vid_info = vid_info[filtered_ids]

        classes = sorted(list(dic_opp.keys()))
        self.classes = classes

        vid_info = vid_info.sort_values("file")
        self.labels = [i.split(',') for i in vid_info[label_key]]

        self.percent = percent
        if percent is not None:
            logger.info(f"Using {percent}% of training data.")
            filtered_vid_info = select_k(vid_info, percent, classes, label_key, dic_opp, label_key, opposite=sample_opposite, seed=seed)
            vid_info = filtered_vid_info.sort_values("file")

        # define transforms
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        if transform:
            self.transform = Compose([
                RandomResizedCrop(128, scale=(0.8, 1.0)),
                RandomApply(ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), p=0.5),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                normalize
            ])
        else:
            self.transform = Compose([Resize((128, 128)), ToTensor(), normalize])
        
        # ensure well defined order
        self.labels = [i.split(',') for i in vid_info[label_key]]
        self.track_files = vid_info[path_key].values

        self.mlb = MultiLabelBinarizer(classes=classes)
        self.vectorised_labels = self.mlb.fit_transform(self.labels)
        
        if self.percent is not None:
            self.label_type = np.array(list(vid_info['flag']))
        
        self.class2num = {j: i for i, j in enumerate(classes)}
        self.num2pos2numneg = {self.class2num[i]: [self.class2num[k] for k in j] for i, j in dic_opp.items()}
        self.weights = self.vectorised_labels.copy()
        for i, j in self.num2pos2numneg.items():
            indices = self.weights[:, i] == 1
            for ind in j:
                self.weights[indices, ind] = 1
        

        self.vid_info = vid_info
        logger.info(f"loaded {split} dataset: {len(self.track_files)} data points")

    def __len__(self):
        return len(self.track_files)

    def __getitem__(self, item):
        img = Image.open(self.filepath_map[self.track_files[item]])
        img = self.transform(img)
        
        clss = self.vectorised_labels[item]
        if self.percent is not None:
            ltype = self.label_type[item]
            return {'img': img, 'label': clss, 'type': ltype}

        return {'img': img, 'label': clss, 'type': 0, 'weights': self.weights[item]}


if __name__ == "__main__":
    filters = {'object': ['fridge', 'drawer', 'bottle', 'spoon', 'fork', 'knife', 'jar']}

    dic_opp = {
            'open': ['close'],
            'close': ['open'],
            'inhand': ['outofhand'],
            'outofhand': ['inhand'],
            'peeled': ['unpeeled'],
            'unpeeled': ['peeled'],
            'whole': ['cut'],
            'cut': ['whole'],
            'cooked': ['raw'],
            'raw': ['cooked'],
    }

    data = StateDataLoader(
        split="test",
        dic_opp=dic_opp,
        sample_opposite=True,
        filter={'object': ['fridge', 'drawer', 'bottle', 'spoon', 'fork', 'knife', 'jar', 'onion', 'egg', 'potato']},
        percent=None,
        seed=0
    )
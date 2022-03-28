import os, sys
sys.path.append("../")
import logging
import pandas as pd
from PIL import Image
from torchvision.transforms.transforms import Resize
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, ColorJitter, Normalize, RandomHorizontalFlip, RandomAffine, RandomErasing, RandomResizedCrop
from io_utils import *
import numpy as np
from sklearn import preprocessing
from data.Base import EpicHandTracks4SimCLR

logger = logging.getLogger("main.data")

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    dataset = worker_info.dataset
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    dataset.rng = np.random.default_rng((torch.initial_seed() + worker_id) % np.iinfo(np.int32).max)

def select_k(frame, k, classes, column, seed=0):
    pos_list = []
    frame = frame.copy()
    for klass in classes:
        labels = [lab for lab in frame[column]]
        out = frame[[klass==i for i in labels]].copy()
        sampled = out.sample(k, random_state=seed, replace=False)
        pos_list.append(sampled.iloc[:k])

    pos_frame = pd.concat(pos_list, axis=0)

    filtered_vid_info = pos_frame
    return filtered_vid_info


class EpicHandTrackswGUN71(EpicHandTracks4SimCLR):

    def __init__(self, config, split='train', transform=False):
        EPIC_split = "validation" if "val" in split else split
        super(EpicHandTrackswGUN71, self).__init__(config['data']['EPIC'], split=EPIC_split)

        key = config['data']['key']
        base_dir = config['data']['dir']
        filter = config['data'].get('filter', None)

        logger.info(f"Key : {key}")
        logger.info(f"Base dir : {base_dir}")
        logger.info(f"Filter : {filter}")

        self.data_dir = os.path.join(base_dir, "data")
        self.split = split

        vid_info = pd.read_pickle(os.path.join(base_dir, "annotations_" + split + "_extra.pkl"))
        # filter files according to criteria
        if filter is not None:
            filtered_ids = [vid_info[k].isin(val).values for k, val in filter.items()]
            filtered_ids = filtered_ids[0] if len(filtered_ids) == 1 else np.logical_and(*filtered_ids)
            vid_info = vid_info[filtered_ids]

        series = vid_info[key].value_counts().sort_index()
        self.classes = list(series.index)

        vid_info = vid_info.sort_values("file")

        series = vid_info[key].value_counts().sort_index()
        self.classes = list(series.index)
        self.weights = np.array(list(series.values))
        
        self.weights = np.sum(self.weights) / (self.weights + 1e-5)
        self.weights = self.weights * len(self.weights) / (np.sum(self.weights) + 1e-3)
        # define transforms
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        if transform:
            s = 1.0
            self.transform_hand = Compose([
                # ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
                RandomResizedCrop((128, 128), scale=(0.5, 1.0)),
                # RandomAffine(30, (0.1, 0.1)),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                normalize
            ])
        else:
            self.transform_hand = Compose([Resize((128, 128)), ToTensor(), normalize])
        
        self.labels = [i for i in vid_info[key]]
        self.images = [os.path.join(self.data_dir, i) for i in vid_info['file']]

        self.mlb = preprocessing.LabelEncoder()
        self.mlb.fit(self.classes)
        self.encoded_labels = self.mlb.transform(self.labels)
        
        logger.info(f"loaded {split} dataset: {len(self.images)} data points")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = Image.open(self.images[item])
        img = self.transform_hand(img)
        
        clss = self.encoded_labels[item]
        
        track_item = self.rng.integers(0, len(self.lengths))
        track_len = self.lengths[track_item]
        window = min(self.window, track_len // 4)
        x, y = ((torch.rand(2) * window).long() +
                (torch.rand(1) * (track_len - window)).long()).tolist()
        
        img_list = get_trajectory_atindices_fromfolder(self.track_folders[track_item], [x, y])
        if self.right_only and ("_l" == self.track_folders[track_item][-2:]):
            frame1 = self.transform_flip(img_list[0])
            frame2 = self.transform_flip(img_list[1])
        else:
            frame1 = self.transform(img_list[0])
            frame2 = self.transform(img_list[1])

        return {'img': img, 'label': clss, "f1": frame1, "f2": frame2}
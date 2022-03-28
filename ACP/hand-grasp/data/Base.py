import os, sys
sys.path.append("../")
import logging
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from io_utils import *
import numpy as np
from transforms import get_simclr_transform

logger = logging.getLogger("main.data")

MAX_TRACK_LEN = 32

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    dataset = worker_info.dataset
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    dataset.rng = np.random.default_rng((torch.initial_seed() + worker_id) % np.iinfo(np.int32).max)

class ABCCollator(object):
    def __init__(self, split):
        self.split = split

    def __call__(self, data):
        imgs = torch.cat([d[0] for d in data], dim=0)
        membership = torch.cat([i * torch.ones(d[1], dtype=torch.long) for i, d in enumerate(data)])
        labels = torch.cat([d[2] for d in data])
        track_len = torch.cat([d[1] * torch.ones(1, dtype=torch.long) for i, d in enumerate(data)])

        out = {"img": imgs, "membership": membership, "label": labels, "track_len": track_len}
        return out


class EpicHandTracksSplit(Dataset, ABC):
    def __init__(self, config, split='train'):
        # get attributes
        subset = config.get('filter', None)
        self.imsize = config.get('imsize', None)
        self.min_track_len = config.get('min_track_len', None)
        self.right_only = config.get('right_only', None)

        logger.info(f"subset : {subset}")
        logger.info(f"imsize : {self.imsize}")
        logger.info(f"min track len : {self.min_track_len}")
        logger.info(f"right_only : {self.right_only}")
        
        base_dir = config['EPIC_dir']
        if base_dir is None or base_dir == "None":
            base_dir = "/data01/mohit/Track-Hands/output/hoa_tracks-mohitthreshold_split/"
        
        logger.info(f"EPIC Tracks dir: {base_dir}")

        # get list of files
        self.base_dir = os.path.join(base_dir, split, "hand")
        self.split = split
        self.rng = np.random.default_rng(0)

        vid_info = pd.read_pickle(os.path.join(base_dir, f"annotations_{split}.pkl"))
        test_s1 = pd.read_csv(f"{base_dir}/EPIC_test_s1_object_video_list.csv")
        test_s2 = pd.read_csv(f"{base_dir}/EPIC_test_s2_object_video_list.csv")
        to_be_removed_videos = list(test_s1['video_id']) + list(test_s2['video_id'])
        all_videos = list(pd.read_csv(f"{base_dir}/EPIC_55_annotations.csv")['video_id']) + to_be_removed_videos
        
        vid_info = vid_info[vid_info['video_id'].isin(all_videos)]
        
        test_particips = config['test_p']
        logger.info(f"Filtering p {test_particips}")

        vid_info = vid_info[~vid_info['participant_id'].isin(test_particips)]
        vid_info = vid_info[vid_info["length"] > self.min_track_len]

        # filter files according to criteria
        if subset is not None:
            filtered_ids = [vid_info[key].isin(val).values for key, val in subset.items()]
            filtered_ids = filtered_ids[0] if len(filtered_ids) == 1 else np.logical_and(*filtered_ids)
            vid_info = vid_info[filtered_ids]
        
        # ensure well defined order
        vid_info = vid_info.sort_values("file")
        self.track_folders = [os.path.join(self.base_dir, os.path.splitext(os.path.basename(f))[0]) for f in list(vid_info["file"])]
        self.lengths = [l for l in list(vid_info["length"])]


        logger.info(f"loaded {split} dataset: {len(self.track_folders)} tracks")

    def extract_segment_from_track(self, num_images):
        track_len = num_images
        if track_len <= MAX_TRACK_LEN:
            return 0, track_len
        idx = self.rng.integers(0, track_len - MAX_TRACK_LEN)
        return idx, idx + MAX_TRACK_LEN

        
    def __len__(self):
        return len(self.track_folders)

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

class EpicHandTracks4SimCLR(EpicHandTracksSplit):

    def __init__(self, config, split="train"):
        super(EpicHandTracks4SimCLR, self).__init__(config, split=split)
        self.transform = get_simclr_transform(s=1, imsize=self.imsize)
        if self.right_only:
            self.transform_flip = get_simclr_transform(s=1, imsize=self.imsize, alwaysflip=True)
        self.window = config.get('window', None)
        logger.info(f"window : {self.window}")

    def __getitem__(self, item):
        track_len = self.lengths[item]
        window = min(self.window, track_len // 4)
        x, y = ((torch.rand(2) * window).long() +
                (torch.rand(1) * (track_len - window)).long()).tolist()
        
        img_list = get_trajectory_atindices_fromfolder(self.track_folders[item], [x, y])
        if self.right_only and ("_l" == self.track_folders[item][-2:]):
            frame1 = self.transform_flip(img_list[0])
            frame2 = self.transform_flip(img_list[1])
        else:
            frame1 = self.transform(img_list[0])
            frame2 = self.transform(img_list[1])

        out = {"f1": frame1, "f2": frame2}
        
        return out

if __name__ == "__main__":

    dataloader = EpicHandTracks4SimCLR(right_only=True)

    # for ind, im in enumerate(dataloader):
    #     print(im["f1"].shape)
    #     if ind == 5:
    #         break
import glob2
import os, random
import logging
from pathlib import Path
import pdb

from PIL import Image
import pandas as pd
import pickle

import torch
from torch import distributions
from torchvision import transforms
import torchvision.transforms.functional as TF

from .transforms import get_simclr_transform
from .data_utils import participant_id_splits
import utils

class EpicFrames:
    def __init__(self, base_dir, split='train', crop="center", max_track_len=128, temporal_simclr=False):
        # get list of files
        self.split = split
        self.temporal_simclr = temporal_simclr
        self.rgb_frames = "/data01/rgoyal6/datasets/EPIC-KITCHENS/rgb_frames"
        vid_info = pd.read_csv(base_dir / f'epic-kitchens-100-annotations/EPIC_100_{split.split("_")[0]}.csv')
        vfpst = pd.read_csv(base_dir / f'epic-kitchens-100-annotations/EPIC_100_video_info.csv')

        # split_pids = participant_id_splits[split]
        split_videos = set(vid_info.video_id)
        # pdb.set_trace()
        # split_videos = {sv for sv in split_videos if any(pid in sv for pid in split_pids)}
        video_lens = {r.video_id: r.duration * r.fps for i, r in vfpst.iterrows() if r.video_id in split_videos}
        self.videos = [v for v in glob2.glob(f"{self.rgb_frames}/*/*") if os.path.basename(v) in split_videos]
        self.video_lens = [video_lens[os.path.basename(v)] for v in self.videos]
        self.categorical = distributions.categorical.Categorical(probs=torch.tensor(self.video_lens))
        num_frames = sum(self.video_lens)
        # filter files according to criteria
        utils.global_logger.info(f"loaded {split} dataset: {num_frames} frames")
        # logger.info(split_videos)

        # define transforms
        self.transform = get_simclr_transform()
        self.max_track_len = max_track_len
        self.crop = crop
        if crop == "bbox":
            with open(os.path.join(base_dir, f'epic-kitchens-100-annotations/EPIC_100_bboxes.pkl'), "rb") as f:
                self.bboxes = pickle.load(f)

    def __len__(self):
        return 131072 if self.split == "train" else 16384

    def __getitem__(self, item):
        while True:
            video_id = self.categorical.sample()
            video_path = self.videos[video_id]
            if self.crop == "bbox":
                base_frame = random.choice(list(self.bboxes[os.path.basename(video_path)].keys()))
                frames = [base_frame,
                          base_frame + int(torch.rand(1).item() * self.max_track_len - self.max_track_len // 2)]
                coords = random.choice(self.bboxes[os.path.basename(video_path)][base_frame])
            else:
                frames = ((torch.rand(2) * self.max_track_len) +
                          (torch.rand(1) * self.video_lens[video_id]) - self.max_track_len).long().tolist()
            paths = [os.path.join(video_path, f"frame_{str(i).zfill(10)}.jpg") for i in frames]
            if False in [os.path.exists(p) for p in paths]:
                continue
            try:
                frames = [Image.open(p) for p in paths]
                break
            except Exception as e:
                print(e, paths)
                continue

        w, h = frames[0].size
        if self.crop == "random":
            side = (torch.rand(1).item() * 0.6 + 0.15) * min(w, h)
            top = torch.rand(1).item() * (h - side)
            left = torch.rand(1).item() * (w - side)
        elif self.crop == "center":
            side = min(w, h)
            top, left = (h - side) // 2, (w - side) // 2
        elif self.crop == "bbox":
            side = min((coords[2] - coords[0]) * w, (coords[3] - coords[1]) * h) * 1.2
            cy, cx = (coords[2] + coords[0]) / 2 * w,  (coords[3] + coords[1]) / 2 * h
            top, left = cx - side / 2, cy - side / 2
        else:
            raise NotImplementedError
        crops = []
        for frame in frames:
            crops.append(TF.resized_crop(frame, top=top, left=left, height=side, width=side, size=[128, 128],
                                         interpolation=transforms.InterpolationMode.BICUBIC))
        crops = [self.transform(crop) for crop in crops]
        return crops
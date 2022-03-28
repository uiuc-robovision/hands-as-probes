import time
import functools
import os, sys
from pathlib import Path
import pickle
import argparse
import pandas as pd
import glob2
import random
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

import torch
from torch import distributions
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

sys.path.append(str(Path(__file__).absolute().parent.parent))
from data.transforms import get_simclr_transform
from data.data_utils import participant_id_splits
from utils import set_torch_seed
from paths import *

class SimpleTracks():
    def __init__(self, split='train', crop="center", max_track_len=128, temporal_simclr=True):
        # get list of files
        self.split = split
        self.temporal_simclr = temporal_simclr
        self.output_dir = DIR_TRACKS / f"{split}_simple{crop}" 
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.rgb_frames_dir = "/data01/rgoyal6/datasets/EPIC-KITCHENS/rgb_frames"
        vid_info = pd.read_csv(f'{DIR_ANNOTATIONS}/EPIC_100_{split.split("_")[0]}.csv')
        self.vfpst = pd.read_csv(f'{DIR_ANNOTATIONS}/EPIC_100_video_info.csv')

        split_videos = {sv for sv in set(vid_info.video_id) if sv not in participant_id_splits[split]}
        video_lens = {r.video_id: r.duration * r.fps for i, r in self.vfpst.iterrows() if r.video_id in split_videos}
        self.videos = [v for v in glob2.glob(f"{self.rgb_frames_dir}/*/*") if os.path.basename(v) in split_videos]
        self.video_lens = [video_lens[os.path.basename(v)] for v in self.videos]
        self.categorical = distributions.categorical.Categorical(probs=torch.tensor(self.video_lens))
        num_frames = sum(self.video_lens)
        print(f"loaded {split} dataset: {num_frames} frames")

        # define transforms
        self.transform = get_simclr_transform()
        self.max_track_len = max_track_len
        self.crop = crop
        if crop == "bbox":
            with open(f'{DIR_ANNOTATIONS}/EPIC_100_bboxes.pkl', "rb") as f:
                self.bboxes = pickle.load(f)
        
    def __len__(self):
        return 131072 if self.split == "train" else 16384

    def __getitem__(self, item):
        while True:
            video_id = self.categorical.sample()
            video_path = self.videos[video_id]
            if self.crop == "bbox":
                base_frame = random.choice(list(self.bboxes[os.path.basename(video_path)].keys()))
                frames = [base_frame, base_frame + int(torch.rand(1).item()*self.max_track_len - self.max_track_len//2)]
                coords = random.choice(self.bboxes[os.path.basename(video_path)][base_frame])
            else:
                frames = ((torch.rand(2) * self.max_track_len) +
                          (torch.rand(1) * self.video_lens[video_id]) - self.max_track_len).long().tolist()

            if self.temporal_simclr:
                vID = Path(video_path).stem
                sample_rate = 5 if self.vfpst[self.vfpst.video_id == vID].iloc[0].fps == 50 else 6
                frames = list(range(min(frames), max(frames) + 1, sample_rate))

            paths = [Path(video_path) / f"frame_{str(i).zfill(10)}.jpg" for i in frames]
            if False in [p.exists() for p in paths]:
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
        return (paths, crops)

    def save_crops(self, data):
        paths, crops = data
        output_path = self.output_dir / f"{paths[0].parent.stem}_{paths[0].stem}_{paths[-1].stem}.jpg"
        new_img = Image.new("RGB", (crops[0].size[0]*len(crops), crops[0].size[1]))
        x_offset = 0
        for im in crops:
            new_img.paste(im, (x_offset,0))
            x_offset += im.size[0]
        new_img.save(output_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", "-c", choices=["center", "random", "bbox"], default="bbox")
    ap.add_argument("--split", "-s", choices=["train", "validation", "test", "all"], default="train")
    args = ap.parse_args()

    splits = [args.split]
    if args.split == 'all':
        splits = ["train", "validation", "test"]
    for split in splits:
        set_torch_seed(0)
        st = SimpleTracks(split=split, crop=args.crop)
        for i in tqdm(range(len(st)), dynamic_ncols=True, desc=split):
            st.save_crops(st[i])
            # next(iter(st))
            # r = tqdm(p.imap(save_crops, iter(st)), total=len(st))
        # for paths, crops in tqdm(st, dynamic_ncols=True, desc=split):
        #     st.save_crops(paths, crops)


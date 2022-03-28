import os, sys, pdb
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

from .epic_tracks import EpicTracks
from .transforms import SimClrTransform, get_simclr_transform
import utils

class SimCLRwTCNCollator(object):
    def __init__(self):
        self.initialized = True

    def __call__(self, data):
        # data: [bs, 2 (attract_pair_1, attract_pair_2), 2 (frame1_{1,2}, frame2_{1,2}), 3, 128, 128]
        # imgs1: [bs * 2 (stack of frame1_1 + frame_2_1), 3, 128, 128] (attract_pair_1)
        # imgs2: [bs * 2 (stack of frame1_2 + frame_2_2), 3, 128, 128] (attract_pair_2)
        imgs1 = torch.cat([d[0] for d in data if d[0][1].any()], dim=0) # cat([(2, 3, 128, 128), (2, 3, 128, 128)], dim=0
        imgs2 = torch.cat([d[1] for d in data if d[1][1].any()], dim=0)
        return imgs1, imgs2

class EpicTracks4SimCLRwTCN(EpicTracks):
    def __init__(self, split='train', subset=None, segment_len_range=(10, 32)):
        super(EpicTracks4SimCLRwTCN, self).__init__(split=split, subset=subset, segment_len_range=segment_len_range)
        self.simclr_transform = get_simclr_transform(s=1)

    def __getitem__(self, item):
        track_path = self.track_files[item]
        segment_len = self.track_lengths[track_path]
        window = 0
        
        x1 = torch.randint(0, segment_len - 3*window, size=(1,)).item()
        y1 = x1 #+ torch.randint(1, window, size=(1,)).item()
        x2 = torch.randint(y1 + window, segment_len - window, size=(1,)).item()
        y2 = x2 #+ torch.randint(1, window, size=(1,)).item()

        # assert(abs(x1 - y1) < window)
        # assert(abs(x2 - y2) < window)
        # assert(abs(max(x1, y1) - min(x2, y2)) >= window)

        frame1_1 = self.simclr_transform(self.extract_image_from_track(item, x1))
        frame1_2 = self.simclr_transform(self.extract_image_from_track(item, y1))
        frame2_1 = self.simclr_transform(self.extract_image_from_track(item, x2))
        frame2_2 = self.simclr_transform(self.extract_image_from_track(item, y2))
        return torch.stack([frame1_1, frame2_1]), torch.stack([frame1_2, frame2_2])
    

class EpicTracks4SimCLRwTCN_Multi_UniformSample(EpicTracks):
    def __init__(self, base_dir, split='train', subset=None, segment_len_range=(10, 32), two_heads=False, vis=False):
        super(EpicTracks4SimCLRwTCN_Multi_UniformSample, self).__init__(base_dir, split=split, subset=subset, segment_len_range=segment_len_range)
        self.vis = vis
        self.two_heads = two_heads
        self.deterministic_transform = True
        self.random_simclr_transform = SimClrTransform(deterministic=False, s=1).apply
        if self.deterministic_transform:
            utils.global_logger.log_once("Using temporally consistent transforms!")
        else:
            self.negatives_simclr_transform = SimClrTransform(deterministic=False, s=1).apply

        # self.negative_ranges = [1, 1.75, 2.5, 3.25, 4]
        # self.negative_ranges = [0, 0, 0, 0, 0]
        self.negative_ranges = [1, 1.5, 2, 2.5, 3]
        # self.negative_ranges = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25]
        utils.global_logger.log_once(f"Negative margin range: {self.negative_ranges}")

    def __getitem__(self, item):
        if self.deterministic_transform:
            self.negatives_simclr_transform = SimClrTransform(deterministic=True, s=1).apply
        track_path = self.track_files[item]
        segment_len = self.track_lengths[Path(track_path).stem]
        window_positive = max(segment_len // 4, 2)

        rand_pts = torch.randperm(window_positive)[:2]
        # rand_pts = torch.tensor([0, window_positive - 1])
        x1, y1 = rand_pts + torch.randint(0, segment_len - rand_pts.max(), size=(1,))
        x1, y1 = x1.item(), y1.item()

        dicts = {}
        dicts["vid"] = self.video_ids["_".join(self.track_files[item].split("_")[:2])]
        x1_img = self.extract_image_from_track(item, x1)
        dicts["pos_1"] = self.random_simclr_transform(x1_img)
        dicts["pos_2"] = self.random_simclr_transform(self.extract_image_from_track(item, y1))
        if self.two_heads:
            dicts["pos_1_2"] = self.random_simclr_transform(x1_img)

        if self.vis:
            imgs = [self.extract_image_from_track(item, i) for i in range(segment_len)]
            dicts["seg"] = transforms.ToTensor()(np.concatenate(imgs, axis=0))
            dicts["pos_1_coord"] = x1
            dicts["pos_2_coord"] = y1
            dicts["pos_1_raw"] = transforms.ToTensor()(self.extract_image_from_track(item, x1))
            dicts["pos_2_raw"] = transforms.ToTensor()(self.extract_image_from_track(item, x1))

        dicts["short_tracks"] = False
        if segment_len < 8:
            assert False
            for i in range(len(negative_ranges)):
                dicts[f"negs_{i}"] = dicts["pos_1"]
                dicts["short_tracks"] = True
            return dicts

        # window_negative = int(1.75*window_positive) # Try 1.25, 1.75, 2.0, 2.5
        # for i in range(5):
        for i, window_negative in enumerate([int(i*window_positive) for i in self.negative_ranges]):
            left_neg, right_neg = None, None
            if (x1 - window_negative) > 0:
                left_neg = torch.randint(0, x1 - window_negative, size=(1,)).item()
            if (x1 + window_negative < segment_len):
                right_neg = torch.randint(x1 + window_negative, segment_len, size=(1,)).item()

            x2 = None
            if left_neg is None and right_neg is None:
                x2 = 0 if torch.rand(1) > 0.5 else segment_len - 1
            elif left_neg is not None and right_neg is not None:
                x2 = left_neg if torch.rand(1) > 0.5 else right_neg
            elif left_neg is not None:
                x2 = left_neg
            else:
                x2 = right_neg

            frame_neg = self.negatives_simclr_transform(self.extract_image_from_track(item, x2))
            dicts[f"negs_{i}"] = frame_neg
            if self.vis:
                dicts[f"negs_{i}_coord"] = x2

        # dicts["oob"] = oobs
        return dicts
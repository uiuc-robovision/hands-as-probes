import os
from pathlib import Path
import logging
import random
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from PIL import Image
import pickle as pkl
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from tqdm.auto import tqdm

from paths import DIR_ANNOTATIONS, DIR_SHAN_DETECTIONS, DIR_TRACKS, DIR_TRACKS_HAND

from .transforms import IM_DIM, get_track_transform
from .data_utils import bb_intersection_over_union, load_metadata
from epic_kitchens.hoa import load_detections

logger = logging.getLogger("main.data")

class EpicKitchenObjectCrops(Dataset, ABC):
    def __init__(self, split='train', iou_threshold=1.0, tracks_hand=False):
        self.split = split

        self.data_suffix = split.split("_", 1)[1]
        root = DIR_TRACKS_HAND if tracks_hand else DIR_TRACKS
        self.tracks_dir = root / self.data_suffix / "images"
        df = load_metadata(*split.split("_", 1))
        tracks = [self.tracks_dir / tf for tf in df.track_name]
        self.object_crops = []
        for t in tqdm(tracks, leave=False, desc=f"Gathering object crop paths ({self.split})"):
            self.object_crops += [(t, i) for i in range(Image.open(t).size[0] // 128)]

        if "train" in self.split and iou_threshold < 1 and self.data_suffix == "ioumf":
            file_ious = self.get_ious()
            logger.info(f"Filtering images based on IoU Threshold: {iou_threshold}")
            self.object_crops = list(filter(lambda f: file_ious.loc[str(f), "iou"] < iou_threshold, self.object_crops))
            logger.info(f"Filtered!")
        
        logger.info(f"loaded {self.split} dataset: {len(self.object_crops)} images")

        # ensure well defined order
        self.object_crops.sort()

    def get_ious(self):
        iou_path = DIR_TRACKS / "metadata" / f"{self.split}_object_crop_ious.pkl"
        if iou_path.exists():
            return pd.read_pickle(iou_path)
        iou_path.parent.mkdir(exist_ok=True, parents=True)

        logger.info(f"Loading image bounding boxes...")
        vid_info = pd.read_csv(DIR_ANNOTATIONS / "EPIC_100_video_info.csv")
        tracks_data = pd.read_pickle(DIR_ANNOTATIONS / "EPIC_100_train_tracks_ioumf.pkl")
        tracks_data = tracks_data.append(
            pd.read_pickle(DIR_ANNOTATIONS / "EPIC_100_validation_tracks_ioumf.pkl"),
            ignore_index=True
        )
        tracks_data = tracks_data.set_index("track_id").to_dict('index')
        logger.info(f"Loaded!")

        bounding_box_cache = {}
        file_ious = {}
        for f in tqdm(self.object_crops, leave=False, desc="Calculating IoUs"):
            file_ious[str(f)] = self.contains_hands(f, tracks_data, vid_info, bounding_box_cache)
        file_ious = pd.DataFrame.from_dict(file_ious, orient='index', columns=["iou"])
        file_ious.to_pickle(iou_path)
        return file_ious

    def contains_hands(self, path, tracks_data, vid_info, cache) -> int:
        track_name = path.parent.stem
        track_id = "_".join(track_name.split("_")[:-1])
        vID = "_".join(track_id.split("_")[:2])
        split_num = int(track_name.split('_')[-1])
        track = tracks_data[track_id]

        # 256 taken from extract_action_localized_tracks.py:max_track_length
        sample_rate = 5 if vid_info[vid_info.video_id == vID].iloc[0].fps == 50 else 6
        sampled_index = [k + (256*split_num) for k in range(0, track["len"], sample_rate)]

        img_index = int(path.stem.split("_")[-1])
        img_index = sampled_index[img_index]

        if vID not in cache:
            cache[vID] = load_detections(DIR_SHAN_DETECTIONS / vID[:3] / f"{vID}.pkl")
        video_detections = cache[vID]

        frame_id, object_id = track["track"][img_index]

        hand_bboxes = [hb.bbox for hb in video_detections[frame_id].hands]
        if hand_bboxes is None:
            return 0
        object_bbox = video_detections[frame_id].objects[object_id].bbox
        
        ious = np.array([bb_intersection_over_union(hb, object_bbox) for hb in hand_bboxes])
        if ious.shape[0] == 0:
            return 0
        return ious.max()


    def __len__(self):
        return len(self.object_crops)

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

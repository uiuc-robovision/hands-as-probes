import os, pdb
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
from tqdm.auto import tqdm

from epic_kitchens.hoa import load_detections

from .data_utils import IM_DIM, load_metadata, participant_id_splits
from .transforms import get_track_transform
import utils
from paths import *

class EpicTracks(Dataset, ABC):
    def __init__(self, split='train', subset=None, transform_list=(), segment_len_range=(10, 32), ):
        # get list of files
        self.split = split
        self.base_dir = DIR_TRACKS / split
        self.min_segment_len, self.max_segment_len = segment_len_range

        spl, data_suffix = split.split("_", 1)
        df = load_metadata(spl, data_suffix)
        self.base_dir = DIR_TRACKS / data_suffix / "images"
        self.track_files = df.track_name.tolist()
        self.track_lengths = df.set_index('track_name')['length'].to_dict()
        
        self.logger = utils.global_logger 
        self.logger.info(f"loaded {split} dataset: {len(self.track_files)} tracks")

        # define transforms
        self.transform = get_track_transform(transform_list, segment_len_range[1])

        # ensure well defined order
        self.track_files.sort()
        vid_list = ["_".join(tf.split("_")[:2]) for tf in self.track_files]
        self.video_ids = {vid:i for i,vid in enumerate(sorted(list(set(vid_list))))}

        # self.track_lengths = self.get_instance_weights()

        if data_suffix == 'srpn':
            self.logger.info("SKIPPING HALF TRACKS")
            indexes = torch.randperm(len(self.track_files))[:len(self.track_files)//2]
            self.track_files = np.array(self.track_files)[indexes].tolist()
            self.track_files.sort()
            total_crops = np.sum([self.track_lengths[tf] for tf in self.track_files])
            self.logger.info(f"Total crops: {total_crops}")


        # store current image
        self.current_image_path = None
        self.current_image = None


    def remove_small_tracks(self, min_length):
        self.track_files = [t for t in self.track_files if self.track_lengths[t] >= min_length]

    def extract_segment_from_track(self, track_path):
        track_len = self.track_lengths[track_path]
        if track_len <= self.max_segment_len:
            return 0, track_len
        start_idx = torch.randint(0, track_len - self.max_segment_len, size=(1,)).item()
        return start_idx, start_idx + self.max_segment_len

    def extract_image_from_track(self, track_idx, image_index):
        base_path = Path(self.base_dir) / self.track_files[track_idx]
        if base_path.is_file():
            if self.current_image_path != base_path:
                self.current_image = Image.open(base_path)
                self.current_image_path = base_path
            return self.current_image.crop((IM_DIM*image_index, 0, (image_index+1)*IM_DIM, IM_DIM))
        image_path = base_path / f"{base_path.stem}_{image_index}.jpg"
        return Image.open(image_path)

    def sample_tracks(self, numtracks=12800):
        self.track_files = random.sample(self.track_files, min(numtracks, len(self.track_files)))
        self.track_files.sort()

    def __len__(self):
        return len(self.track_files)

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

import os
import pdb
from pathlib import Path
from PIL import Image

import torch

from .epic_tracks import EpicTracks
from .data_utils import IM_DIM

class EpicTracks4TCN(EpicTracks):
    def __init__(self, split, subset, transform_list=(), segment_len_range=()):
        super(EpicTracks4TCN, self).__init__(split, subset, transform_list, segment_len_range)

    def __getitem__(self, item):
        '''
        window = 12 
        margin = 6
        i1 = np.random.randint(1, track_len - 1)
        si2 = set(np.arange(max(0, i1 - window // 2), min(track_len - 1, i1 + window // 2) + 1)) - {i1}
        i2 = np.random.choice(list(si2))

        s1 = set(np.arange(0, max(0, i1 - window // 2 - margin) + 1))
        s2 = set(np.arange(min(track_len - 1, i1 + window // 2 + margin), track_len))
        si3 = s1.union(s2) - {i2}
        i3 = np.random.choice(list(si3))
        '''
        track_path = self.track_files[item]
        start_idx, end_idx = self.extract_segment_from_track(track_path)
        flag = torch.rand(1).item() > 0.5
        track_len = end_idx - start_idx
        window = track_len // 4
        i1, i2 = (torch.rand(2) * window + (track_len - window if flag else 0)).long().tolist() 
        i3 = int(torch.rand(1).item() * window) + (0 if flag else track_len - window)
        
        img1 = self.transform(self.extract_image_from_track(item, start_idx + i1)).squeeze()
        img2 = self.transform(self.extract_image_from_track(item, start_idx + i2)).squeeze()
        img3 = self.transform(self.extract_image_from_track(item, start_idx + i3)).squeeze()

        return img1, img2, img3
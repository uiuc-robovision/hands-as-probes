from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset

from .data_utils import IM_DIM
from .transforms import SimClrTransform
from .et_simclr import EpicTracks4SimCLR
import utils
from paths import *

def get_normalized_coords(object, hand):
        width = object.bbox.right - object.bbox.left
        height = object.bbox.bottom - object.bbox.top
        object_cx = (object.bbox.left + object.bbox.right) / 2
        object_cy = (object.bbox.top + object.bbox.bottom) / 2
        hand_cx = (hand.bbox.left + hand.bbox.right) / 2
        hand_cy = (hand.bbox.top + hand.bbox.bottom) / 2
        posx = (object_cx - hand_cx) / width
        posy = (object_cy - hand_cy) / height
        scalex = (hand.bbox.right - hand.bbox.left) / width
        scaley = (hand.bbox.bottom - hand.bbox.top) / height
        return torch.tensor([posx, posy, scalex, scaley])

def load_hand_metadata(split):
    print("loading metadata")
    df = pd.read_pickle(Path(DIR_TRACKS) / 'ioumf' / f'track_detections_{split}.pkl')
    df.reset_index(inplace=True)
    print("loaded!")
    return df

class EpicHandObjectsTracks(Dataset):
    def __init__(self, split, config, segment_len_range):
        super(EpicHandObjectsTracks, self).__init__()
        global tracks_det
        
        assert "ioumf" in split, "Only IOUMF tracks supported"
        self.logger = utils.global_logger
        self.split = split
        self.config = config
        filter_tsc_on_hands = config["data"]["filter_on_hands"]

        min_oh_len = config["data"]["minlen_hands"]

        splt, data_suffix = split.split("_", 1)
        df = load_hand_metadata(splt)
        self.metadata = df.set_index('track_name').to_dict(orient='index')

        tracks_dir_objects = DIR_TRACKS / data_suffix / "images"
        tracks_dir_hands = DIR_TRACKS_HAND / data_suffix / "images"
        self.tf_objects = [tracks_dir_objects / tf for tf in sorted(df.track_name)]
        self.tf_hands = [tracks_dir_hands / tf for tf in sorted(df.track_name)]
        self.tsc_dataset = EpicTracks4SimCLR(split, None, segment_len_range)
        
        self.logger.info(f"Hand-Objects: Loaded {split} hand-object set: {len(self.tf_objects)} tracks")
        self.tf_objects = [tf for tf in self.tf_objects if self.metadata[tf.name]['hand_valid_indices'].shape[0] >= min_oh_len]
        self.tf_hands = [tf for tf in self.tf_hands if self.metadata[tf.name]['hand_valid_indices'].shape[0] >= min_oh_len]
        self.logger.info(f"Hand-Objects: After filtering {split} tracks with <{min_oh_len} hands: {len(self.tf_objects)}")

        self.logger.info(f"Objects: Loaded {split} TSC set: {len(self.tsc_dataset)} tracks")
        self.tsc_dataset.remove_small_tracks(segment_len_range[0])
        self.logger.info(f"Objects: After filtering {split} with |track| <{segment_len_range[0]}: {len(self.tsc_dataset)}")
        if filter_tsc_on_hands:
            self.tsc_dataset.track_files = [tf for tf in self.tsc_dataset.track_files if self.metadata[Path(tf).name]['hand_valid_indices'].shape[0] >= min_oh_len]
            self.logger.info(f"Objects: After filtering {split} with |track| <{min_oh_len} hands: {len(self.tsc_dataset)}")

        self.placeholder_tensor = self.tsc_dataset.simclr_transform(Image.new(mode="RGB", size=(IM_DIM, IM_DIM)))
        self.hand_tsc_transform = SimClrTransform(deterministic=False, s=1, allow_hflip=False)


    def extract_image_from_track(self, track, image_index):
        return track.crop((IM_DIM*image_index, 0, (image_index+1)*IM_DIM, IM_DIM))


    def __len__(self):
        return max(len(self.tf_objects), len(self.tsc_dataset))


    def __getitem__(self, index):
        tsc_dict = self.tsc_dataset[index]
        object_pos1 = tsc_dict["pos1"]
        object_pos2 = tsc_dict["pos2"]

        track_name = self.tf_hands[index].name
        object_track = Image.open(self.tf_objects[index])
        hand_track = Image.open(self.tf_hands[index])
        valid_indices = self.metadata[track_name]["hand_valid_indices"]
        hand_match_window = hand_track.size[0] // IM_DIM // 4
        oh_transform = SimClrTransform(deterministic=True, s=1, allow_hflip=False)

        oho = valid_indices[torch.randint(0, len(valid_indices), size=(1,))].item()
        pos_indices = valid_indices[np.abs(valid_indices - oho) <= hand_match_window]
        ohh = pos_indices[torch.randint(0, len(pos_indices), (1,))].item()
        object_hand_obj = oh_transform(self.extract_image_from_track(object_track, oho))
        object_hand_hand = oh_transform(self.extract_image_from_track(hand_track, ohh))
        
        # hand_2, hand_3 = self.placeholder_tensor, self.placeholder_tensor
        pose_info = 0
        if self.config["training"]["add_hand_motion"]:
            # Concatenate positions for nearby hands
            position_of_ohh = (valid_indices == ohh).nonzero()[0][0]
            h2 = valid_indices[max(0, position_of_ohh - 1)]
            h3 = valid_indices[min(position_of_ohh + 1, valid_indices.shape[0]-1)]
            obj = self.metadata[track_name]["objects"][oho]
            hands = self.metadata[track_name]["hands"]
            ohh_pose = get_normalized_coords(obj, hands[ohh]).unsqueeze(0)
            h2_pose = get_normalized_coords(obj, hands[h2]).unsqueeze(0)
            h3_pose = get_normalized_coords(obj, hands[h3]).unsqueeze(0)
            pose_info = torch.cat([ohh_pose, h2_pose, h3_pose], dim=0)

                    
        return {
            "object_pos1": object_pos1,
            "object_pos2": object_pos2,
            "object_hand_obj": object_hand_obj,
            "object_hand_hand": object_hand_hand,
            "hand_pose": pose_info,
        }
        
# path = Path(__file__).parent / "check" / f"{index}.png"
# path.parent.mkdir(exist_ok=True, parents=True)
# torchvision.utils.save_image([
#     TF.to_tensor(object_pos1), unnormalize_tensor(self.random_simclr_transform(object_pos1)), 
#     TF.to_tensor(object_pos2), unnormalize_tensor(self.random_simclr_transform(object_pos2)),
#     TF.to_tensor(object_hand_pos1), unnormalize_tensor(oh_transform(object_hand_pos1)),
#     TF.to_tensor(object_hand_pos2), unnormalize_tensor(oh_transform(object_hand_pos2))], 
# path)

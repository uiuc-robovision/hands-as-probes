from collections import defaultdict
import os
import pdb
from pathlib import Path
from PIL import Image
import torch
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from paths import *

IM_DIM = 128

# nov05_2021
# Part of train also used in validation
pids_use_in_val = ['P05', 'P22', 'P27', 'P23']
participant_id_splits = {
    "train": ['P01', 'P03', 'P05', 'P06', 'P08', 'P13', 'P17', 'P21', 'P22', 'P23', 'P25', 'P26', 'P27', 'P29'],
    "validation": ['P14', 'P07', 'P04'] + pids_use_in_val,
    "test": ['P10', 'P31', 'P02', 'P28', 'P30', 'P12', 'P24', 'P15', 'P19', 'P20', 'P16'],
}
participant_id_splits_dj = {
    "train": ['P01', 'P03', 'P05', 'P06', 'P08', 'P13', 'P17', 'P21', 'P22', 'P23', 'P25', 'P26', 'P27', 'P29'],
    "validation": ['P14', 'P07', 'P04'],
    "test": ['P10', 'P31', 'P02', 'P28', 'P30', 'P12', 'P24', 'P15', 'P19', 'P20', 'P16'],
}

def get_splits(split=None, disjoint=True):
    source = participant_id_splits
    if disjoint:
        source = participant_id_splits_dj
    if split:
        return source[split]
    return source

def load_metadata(split, data_suffix):
    path = DIR_TRACKS / data_suffix / f'tracks_md_{split}.csv'
    if path.exists():
        return pd.read_csv(path)
    lengths = {}
    for i, p in enumerate(tqdm(os.listdir(path.parent / "images"), leave=False, desc="Caching track lengths")):
        full_path = path.parent / "images" / p
        assert full_path.exists()
        lengths[Path(p).name] = Image.open(full_path).size[0] // IM_DIM
    df = pd.DataFrame.from_dict(lengths, columns=['length'], orient='index')
    df.index.name = 'track_name'
    df.reset_index(inplace=True)
    df["pid"] = [name.split("_")[0] for name in df.track_name]
    
    source = get_splits(disjoint=True)
    for spl, pids in source.items():
        df[df.pid.isin(pids)].to_csv(DIR_TRACKS / data_suffix / f'tracks_md_{spl}.csv', index=False)
    return pd.read_csv(path)


def bb_intersection_over_union(boxA, boxB):
    # Taken from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = [boxA.left, boxA.top, boxA.right, boxA.bottom]
    boxB = [boxB.left, boxB.top, boxB.right, boxB.bottom]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max((xB - xA, 0)) * max((yB - yA), 0)
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
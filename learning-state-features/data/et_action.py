import os
import pdb
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose, ToTensor, ColorJitter
from torchvision import transforms

from data.epic_tracks import EpicTracks
from paths import DIR_ANNOTATIONS, DIR_TRACKS
from .transforms import get_mitstates_transform, TensorJitter, ReshapeTrack, BatchRandomFlip, TemporalSample, TensorNormalize
from .data_utils import participant_id_splits, participant_id_splits_dj
import utils

IM_DIM = 128
MAX_TRACK_LEN = 16

class EpicTracksAction(EpicTracks):
    def __init__(self, base_dir, split='train', transform=False, filter=None, sample=False, frac=0.3):
        super(EpicTracksAction, self).__init__(base_dir, split)
        # get list of files
        self.split = split
        vid_info = pd.read_csv(base_dir / f'epic-kitchens-100-annotations/EPIC_100_{split.split("_")[0]}.csv')
        self.classes = vid_info.verb.unique()
        
        # define transforms
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        if transform:
            self.transform = Compose([
                # ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # RandomHorizontalFlip(p=1.0),
                ToTensor(),
                normalize
            ])

            self.post_transform = Compose([
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # RandomHorizontalFlip(p=0.5),
                ToTensor(),
                normalize
            ])
        else:
            self.transform = Compose([ToTensor(), normalize])

            self.post_transform = Compose([ToTensor(), normalize])
        # ensure well defined order
        self.track_files.sort()

        action_dict = {r.narration_id: r.verb for i, r in vid_info.iterrows()}
        name_to_ann_id = lambda s: "_".join(s.split("_")[i] for i in [0, 1, 3])
        # pdb.set_trace()
        self.track_files = [tf for tf in self.track_files if name_to_ann_id(tf) in action_dict]
        self.logger.info(f"After removing tracks without narrations: {len(self.track_files)} tracks")
        verb_list = [action_dict[name_to_ann_id(tf)] for tf in self.track_files]
        verb2cluster = {v: i for i, v in enumerate(set(verb_list))}
        self.cluster = [verb2cluster[v] for v in verb_list]

        # sample for prototype initialization
        self.sample = sample

        # Fraction used while splitting video on both sides
        self.frac = frac
        self.new_MAX_LENGTH = 2

    def __len__(self):
        return len(self.track_files)

    def __getitem__(self, item):
        path = self.track_files[item]
        track_len = self.track_lengths[path]
        img = np.zeros((track_len, IM_DIM, IM_DIM, 3), dtype=np.uint8)
        for i in range(track_len):
            img[i] = np.array(self.extract_image_from_track(item, i))
        
        # sample 5 frames uniformly.
        if self.frac is not None:
            index = np.ceil(self.frac * len(img))

            i1 = np.random.randint(1, len(img)-1)
            si2 = set(np.arange(max(0, i1-2), min(len(img)-1, i1+2) + 1)) - {i1}
            # i2 = np.random.choice(list(si2))
            
            s1 = set(np.arange(0, max(0, i1-4) + 1))
            s2 = set(np.arange(min(len(img)-1, i1+4), len(img)))
            si3 = s1.union(s2)
            # print(si3, s1, s2)
            i3 = np.random.choice(list(si3))

            img1 = Image.fromarray(img[min(i1, i3)])
            img3 = Image.fromarray(img[max(i1, i3)])

            img1 = self.post_transform(img1)
            img3 = self.post_transform(img3)

            action = self.cluster[item]

            out = {"img1": img1, "img2": img3, "action": action}

            return out


class EpicTracksCleanAction(Dataset):
    def __init__(self, split, noun=True, num_samples=5):
        super(EpicTracksCleanAction, self).__init__()
        self.noun = noun
        self.num_samples = num_samples

        self.df = pd.read_csv("/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/track_info_actions.csv")
        if self.noun:
            self.classes = self.df.noun.unique()
            self.label_encoder = LabelEncoder().fit(self.classes)
        else:
            assert False, "Update code"
            self.df = self.df[(self.df.start_action.str.len() > 0) & (self.df.end_action.str.len() > 0)]
            self.classes = set(self.df.start_action.values)

        pids = participant_id_splits[split]
        self.df = self.df[self.df.participant_id.isin(pids)]
        self.transform = get_mitstates_transform(augment=split=='train')

    def __len__(self):
        return self.df.shape[0]//2
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        images = self.load_trajectory(row.path)
        label = row.noun
        if not self.noun:
            label = list(set([row.start_action, row.end_action]))
        out = {"label": self.label_encoder.transform((label,))[0]}

        old_len = len(images)
        for i in range(old_len, self.num_samples):
            images.append(images[torch.randint(0, old_len, size=(1,))])

        for c, i in enumerate(torch.randperm(len(images))[:self.num_samples]):
            out[f"image_{c}"] = self.transform(images[i])
        return out

    def load_trajectory(self, impath):
        images = Image.open(impath)
        exif = images.getexif()
        num_images = exif[0]
        width = exif[1]
        height = exif[2]
        cols = images.size[0] // exif[1]
        rows = images.size[1] // exif[2]
        all_images = []
        for imnum in range(exif[0]):
            i, j = imnum % cols, imnum // cols
            image = images.crop((i*width, j*height, i*width + width , j*height + height))
            all_images.append(image)
        return all_images

def get_abmil_track_transform(transform_list):
    T = [transforms.ToTensor(), ReshapeTrack(IM_DIM)]
    if "temporal" in transform_list:
        T.append(TemporalSample(MAX_TRACK_LEN))
    if "color" in transform_list or "crop" in transform_list:
        T.append(TensorJitter(color="color" in transform_list, crop="crop" in transform_list))  # , imsize=IM_DIM))
    if "flip" in transform_list:
        T.append(BatchRandomFlip())
    T.append(TensorNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(T)

class ABMILCollator(object):
    def __init__(self):
        self.initialized = True

    def __call__(self, data):
        bI = torch.cat([d["btracks"] for d in data], dim=0)
        aI = torch.cat([d["atracks"] for d in data], dim=0)
        bN = torch.cat([d["bnum"] for d in data])
        aN = torch.cat([d["anum"] for d in data])
        labels = torch.cat([d["labels"] for d in data])
        return {
            "btracks": bI, 
            "atracks": aI, 
            "bnum": bN,
            "anum": aN, 
            "labels": labels
        }

class ETABMIL(Dataset):
    def __init__(self, split='train', filter=None, transform_list=(), verb2class=None):
        # get list of files
        self.base_dir = DIR_TRACKS / "hoa"
        data = pd.read_csv(self.base_dir / "data.csv")
        self.track_files = [Path(p).name for p in data[data.pid.isin(participant_id_splits[split])].path]

        # metadata
        vid_info = pd.read_csv(DIR_ANNOTATIONS / "EPIC_100_train.csv").append(pd.read_csv(DIR_ANNOTATIONS / "EPIC_100_validation.csv"))
        verb_info = pd.read_csv(DIR_ANNOTATIONS / "EPIC_100_verb_classes.csv")
        verb_info = {r.id: r.key for i, r in verb_info.iterrows()}
        self.numstates = None

        # define transforms
        self.transform = get_abmil_track_transform(transform_list)
        # ensure well defined order
        self.track_files.sort()

        # action class labels for top A actions
        action_dict = {r.narration_id: r.verb_class for i, r in vid_info.iterrows()}
        verb_list = [action_dict["_".join(tf.split("_")[:3])] for tf in self.track_files]

        # display verb count
        verb_count = {}
        for v in verb_list:
            if v in verb_count.keys():
                verb_count[v] += 1
            else:
                verb_count[v] = 1
        verb_count = sorted(verb_count.items(), key=lambda k: k[1], reverse=True)[:32]
        disp_verb_count = {verb_info[v]: c for v, c in verb_count}
        utils.global_logger.info(f"Counts: {disp_verb_count}")

        # set verb2class if not provided
        if verb2class is None:
            verb2class = {tup[0]: i for i, tup in enumerate(verb_count)}
            self.weight = 1 / torch.FloatTensor([v[1] for v in verb_count])
        self.pairs = {("_".join(self.track_files[i].split("_")[:-2]), verb2class[v]) for i, v in enumerate(verb_list)
                      if v in verb2class.keys()}
        self.pairs = list(self.pairs)
        self.xy = [(self.track_files[i], verb2class[v]) for i, v in enumerate(verb_list) if v in verb2class.keys()]
        utils.global_logger.info(f"loaded {split} dataset: {len(self.xy)}/{len(self.track_files)} tracks")
        utils.global_logger.info(f"loaded {split} dataset pairs: {len(self.pairs)}")
        self.verb2class = verb2class

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        bI, aI, bN, aN = [], [], [], []
        lab = self.pairs[item][1]
        for sfx in ["_0_0.jpg", "_1_0.jpg"]:
            path = self.base_dir / "data" / (self.pairs[item][0] + sfx)
            if path.exists():
                img = Image.open(path)
                img = self.transform(img)
            else:
                img = torch.zeros(MAX_TRACK_LEN, 3, IM_DIM, IM_DIM)
            bnum, anum = [len(img) // 2] * 2
            aN.append(anum)
            bN.append(bnum)
            bI.append(torch.cat((img[:bnum], torch.zeros(MAX_TRACK_LEN - bnum, 3, IM_DIM, IM_DIM)), dim=0))
            aI.append(torch.cat((img[-anum:], torch.zeros(MAX_TRACK_LEN - anum, 3, IM_DIM, IM_DIM)), dim=0))
            assert bnum + anum == len(img) or anum + bnum == len(img) - 1
        return {
            "btracks": torch.stack(bI), 
            "atracks": torch.stack(aI), 
            "bnum": torch.LongTensor(bN),
            "anum": torch.LongTensor(aN), 
            "labels": torch.LongTensor([lab, lab])
        }
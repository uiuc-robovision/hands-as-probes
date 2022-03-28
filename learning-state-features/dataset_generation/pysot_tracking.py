import os, sys
import argparse
from pathlib import Path

import cv2
import torch
import random
import numpy as np
from glob import glob
from multiprocessing import Pool
import pandas as pd
import PIL
import pdb
from tqdm import tqdm
import functools
import time

sys.path.append(str(Path(__file__).absolute().parent.parent))
from data.data_utils import get_splits
from paths import *

sys.path.append(str(Path(__file__).absolute().parent.parent / 'lib/pysot'))
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from epic_kitchens.hoa import load_detections

# Detectron dependencies
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from lib.detectron2.demo.predictor import VisualizationDemo

MIN_TRACK_LENGTH = 8
SCORE_THRESH = 0.80
SLACK_THRESH = 2
MAX_TRACK_LEN = 256
IM_DIM = 128
HAND_THRESH = 0.5
OBJ_THRESH = 0.4


def extract_crop(bbox, img, dilation=0.2):
    xcenter = bbox[1] + bbox[3] / 2
    ycenter = bbox[0] + bbox[2] / 2
    side = min((max(bbox[2], bbox[3]) / 2) * (1 + dilation), ycenter, xcenter, 256 - xcenter, 456 - ycenter)
    coords = (ycenter - side, xcenter - side, ycenter + side, xcenter + side)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = PIL.Image.fromarray(img, "RGB")
    crop = img.crop(coords).resize((IM_DIM, IM_DIM), PIL.Image.ANTIALIAS)
    return crop


class Detectron:
    SCORE_THRESHOLD = 0.1

    def __init__(self):
        config_path = Path(__file__).parent / "configs" / "mask_rcnn.yaml"
        opts = ['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl']
        confidence_threshold = 0
        cfg = self.setup_cfg(str(config_path), opts, confidence_threshold)
        self.predictor = DefaultPredictor(cfg)
        self.demo = VisualizationDemo(cfg)

    def setup_cfg(self, config_file, opts, confidence_threshold):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.freeze()
        return cfg

    def predict(self, path):
        img = read_image(path, 'BGR')
        boxes, scores, classes = self.demo.pred_on_image(img)

        # get top 10 scores
        valid_i = torch.argsort(torch.tensor(scores), descending=True)[:10]
        boxes = boxes.tensor.cpu().numpy().astype(np.int32)[valid_i]
        # scores = [scores[i] for i in valid_i]
        # classes = [classes[i] for i in valid_i]
        return boxes
    

class Tracker:
    def __init__(self, model, config, hand, track_type, args):
        cfg.merge_from_file(config)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda')
        self.model = ModelBuilder()
        self.model.load_state_dict(torch.load(model,
                                         map_location=lambda storage, loc: storage.cpu()))
        self.model.eval().to(device)
        self.tracker = build_tracker(self.model)
        self.start_idx = 0
        self.track_num = 0
        self.track = []
        self.rgb_frames_dir = DIR_RGB_FRAMES
        fps = pd.read_csv(f"{DIR_ANNOTATIONS}/EPIC_100_video_info.csv")
        self.sr = {x.video_id: round(x.fps / 10) for i, x in fps.iterrows()}
        self.hand = hand
        self.track_type = track_type
        self.init_random_crops = args.random_crops
        self.init_maskrcnn = args.maskrcnn
        self.output_dir = DIR_TRACKS / self.track_type / "images"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        if self.init_maskrcnn:
            self.detectron = Detectron()

        if self.init_random_crops:
            self.box_sizes = None 

    def get_next_detection(self, start, detections, video_id):
        if self.init_random_crops and self.box_sizes is None:
            self.box_sizes = np.array([(int((obj.bbox.bottom - obj.bbox.top))*256, int((obj.bbox.right - obj.bbox.left)*456)) for det in detections for obj in det.objects])

        for idx, det in enumerate(detections[start:]):
            # if self.init_maskrcnn and not (0.1*len(detections) < init_frame_idx < 0.9*len(detections)):
            #     continue
            if len([do for do in det.objects if do.score > OBJ_THRESH]) == 0 or len([dh for dh in det.hands if dh.score > HAND_THRESH]) == 0:
                continue
            hoi = det.get_hand_object_interactions(object_threshold=OBJ_THRESH, hand_threshold=HAND_THRESH)
            if len(hoi) == 0:
                continue
            interacted_obj = [det.objects[v] for k, v in hoi.items() if det.hands[k].side.value == self.hand]
            if len(interacted_obj) == 0:
                continue
            
            init_frame_idx = start + idx + 1
            if self.init_random_crops:
                h,w  = self.box_sizes[np.random.randint(self.box_sizes.shape[0])]
                coords_all = np.array([[o.bbox.left*456, o.bbox.top*256, o.bbox.right*456, o.bbox.bottom*256] for o in interacted_obj], dtype=int)
                bounds = np.array([*coords_all[:, :2].min(0), *coords_all[:, 2:].min(0)])
                indices = [(x,y) for x in range(456) for y in range(256) 
                            if (y+h < bounds[1] or (y > bounds[3] and y+h < 256)) and 
                               (x+w < bounds[0] or (x > bounds[2] and x+w < 456))
                          ]
                if len(indices) == 0:
                    continue
                top_x, top_y = random.choice(indices)
                init_bbox = (top_x, top_y, w, h)
                # print(bounds, init_bbox)
                return init_frame_idx, init_bbox

            elif self.init_maskrcnn:
                # print("checking:", init_frame_idx)
                boxes = self.detectron.predict(self.rgb_frames_dir / video_id[:3] / 'rgb_frames' / video_id / f"frame_{str(init_frame_idx).zfill(10)}.jpg")
                coords_all = np.array([[o.bbox.left*456, o.bbox.top*256, o.bbox.right*456, o.bbox.bottom*256] for o in interacted_obj], dtype=int)
                bounds = np.array([*coords_all[:, :2].min(0), *coords_all[:, 2:].min(0)])
                if boxes.shape[0] == 0:
                    continue
                for i in np.random.permutation(boxes.shape[0]):
                    x1, y1, x2, y2 = boxes[i]
                    # print("testing", (x1, y1, x2, y2), bounds)
                    if (y2 < bounds[1] or y1 > bounds[3]) and (x2 < bounds[0] or x1 > bounds[2]):
                        init_bbox = (x1, y1, x2-x1, y2-y1)
                        return init_frame_idx, init_bbox 
            
            else:
                max_score = 0
                for io in interacted_obj:
                    if io.score > max_score:
                        max_score = io.score
                        bbox_coords = io.bbox
                init_bbox = (bbox_coords.left * 456,
                                bbox_coords.top * 256,
                                (bbox_coords.right - bbox_coords.left) * 456,
                                (bbox_coords.bottom - bbox_coords.top) * 256)
                return init_frame_idx, init_bbox
        
        return None, None

    def get_tracks(self, video_id):
        def get_frames(video_name, sr):
            images = glob(str(self.rgb_frames_dir / video_name[:3] / 'rgb_frames' / video_name / '*.jp*'))
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0][6:]))
            for img in images[::sr]:
                frame = cv2.imread(img)
                yield int(img.split('/')[-1].split('.')[0].split('_')[1]) - 1, frame

        detections = load_detections(Path(DIR_SHAN_DETECTIONS) / video_id[:3] / (video_id + '.pkl'))

        print(video_id, "start:", len(detections))
        self.reset(0, video_id, detections)
        slack = 0
        for idx, fr in get_frames(video_id, self.sr[video_id]):
            if idx < self.start_idx:
                continue
            
            outputs = self.tracker.track(fr)
            if outputs["best_score"] > SCORE_THRESH:
                self.track.append(extract_crop(outputs["bbox"], fr))
                slack = 0
            else:
                slack += 1
                if slack >= SLACK_THRESH:
                    slack = 0
                    self.reset(idx, video_id, detections)
            
            if len(self.track) >= MAX_TRACK_LEN:
                slack = 0
                self.reset(idx, video_id, detections)
            
    def reset(self, start_idx, video_id, detections):
        if len(self.track) > MIN_TRACK_LENGTH:
            self.save(video_id)
        init_frame_idx, init_bbox = self.get_next_detection(start_idx, detections, video_id)
        
        # if no more detections are found
        if init_frame_idx is None:
            return False

        # if image is missing or loading fails
        try:
            # load image corresponding to next detection
            init_frame = cv2.imread(str(self.rgb_frames_dir / video_id[:3] / 'rgb_frames' / video_id / f"frame_{str(init_frame_idx).zfill(10)}.jpg"))
            self.start_idx = init_frame_idx
            self.tracker.init(init_frame, init_bbox)
        except Exception as e:
            print("failed to init tracker, trying again.", video_id, e)
            return self.reset(init_frame_idx + 1, video_id, detections)
        self.track = []

    def save(self, video_id):
        self.track_num += 1
        for s, seg in enumerate(range(0, len(self.track), MAX_TRACK_LEN)):
            segmented_track = self.track[seg: seg + MAX_TRACK_LEN]
            new_im = PIL.Image.new('RGB', (IM_DIM * len(segmented_track), IM_DIM))
            for imnum, im in enumerate(segmented_track):
                new_im.paste(im, (imnum * IM_DIM, 0))
            new_im.save(self.output_dir / f"{video_id}_{self.hand}_{self.track_num}_{s}.jpg")


def mp_wrapper(data_suffix, args, video_id):
    np.random.seed(os.getpid() % 123456789)
    random.seed(os.getpid() % 123456789)
    torch.manual_seed(os.getpid() % 123456789)
    base_path = Path(__file__).absolute().parent.parent / "lib/pysot/experiments"
    for hand in [0, 1]:
        tracker = Tracker(str(base_path / "siamrpn_r50_l234_dwxcorr/model.pth"),
                          str(base_path / "siamrpn_r50_l234_dwxcorr/config.yaml"),
                          hand, data_suffix, args)
                                
        tracker.get_tracks(video_id)
    return True

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--random_crops", action='store_true', help="Initialize with background crops instead of Shan detections.")
    ap.add_argument("--maskrcnn", action='store_true', help="Initialize with maskrcnn object crops instead of Shan detections.")
    ap.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to split data")
    ap.add_argument("--chunk", "-c", type=int, default=0, help="Current chunk index")
    args = ap.parse_args()

    
    data_suffix = 'srpn'
    if args.maskrcnn:
        data_suffix += '_maskrcnn'
    elif args.random_crops:
        data_suffix += '_randomcrops'

    annotations = pd.read_csv(DIR_ANNOTATIONS / "EPIC_100_train.csv").append(pd.read_csv(DIR_ANNOTATIONS / "EPIC_100_validation.csv"))
    all_pids = get_splits()["train"] + get_splits()["validation"]# + get_splits()["test"]
    annotations = annotations[annotations.participant_id.isin(all_pids)]

    unique_videos = list(sorted(annotations.video_id.unique()))

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    
    if args.num_chunks > 1:
        chunk_size = int(np.ceil(len(unique_videos) / args.num_chunks))
        unique_videos = unique_videos[args.chunk*chunk_size:(args.chunk+1)*chunk_size]
        print(unique_videos)

    with Pool(4 if args.maskrcnn else 8) as p:
        wrapped_func = functools.partial(mp_wrapper, data_suffix, args)
        r = list(tqdm(p.imap(wrapped_func, unique_videos), total=len(unique_videos), dynamic_ncols=True, leave=True))

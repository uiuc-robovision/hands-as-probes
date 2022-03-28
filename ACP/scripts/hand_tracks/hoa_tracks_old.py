from cmath import isnan
from epic_kitchens.hoa import load_detections
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from scipy.ndimage import generic_filter
from utils.io_utils import load_trajectory, save_trajectory
from utils.utils import subplot
from absl import app, flags
import multiprocessing
from functools import partial
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'RuntimeWarning: All-NaN slice encountered')

# base_dir = "/data01/mohit/Track-Hands/video-object-states/data/epic-kitchens/metadata"
# data_dir = "/home/mohit/EPIC-KITCHENS"
# hoa_dir = f"{base_dir}/detections/hand-objects"

imsize = 128

FLAGS = flags.FLAGS
flags.DEFINE_string('out_dir', '/data01/mohit/Track-Hands/output/original_tracks', '')
flags.DEFINE_string('data_dir', "/home/mohit/EPIC-KITCHENS", 'Path to EPIC-KITCHENS frames')
flags.DEFINE_string('hoa_dir',
                    "/data01/mohit/Track-Hands/video-object-states/data/epic-kitchens/metadata/detections/hand-objects",
                    'Path to hand-object detections for EPIC dataset')
flags.DEFINE_float('maxhandscore', 0.8, '')
flags.DEFINE_integer('fps', 10, '')
flags.DEFINE_integer('median_filter_size', 2, '')
flags.DEFINE_bool("filter", True, "")
flags.DEFINE_bool("plot_hist", False, "")

METADATA_PATH = "../annotations/"

def make_square(coords, im_width=1, im_height=1):
    widths = coords[:, 2] - coords[:, 0]
    heights = coords[:, 3] - coords[:, 1]
    ycenter = (coords[:, 1] + coords[:, 3])/2
    xcenter = (coords[:, 0] + coords[:, 2])/2
    
    heights = widths = np.maximum(heights, widths)

    out = coords + 0
    out[:, 0] = (xcenter - widths/2).clip(0, im_width)
    out[:, 2] = (xcenter + widths/2).clip(0, im_width)
    out[:, 1] = (ycenter - heights/2).clip(0, im_height)
    out[:, 3] = (ycenter + heights/2).clip(0, im_height)
    return out

def filter_and_dilate_track(coords, scores, window=5, dilation=0.2, im_width=1, im_height=1): 
    """
    track : List[List] ordered [left, top, right, bottom]
    window: window size for median filtering

    returns List[List] ordered [left, top, right, bottom]
    """
    widths = coords[:, 2] - coords[:, 0]
    heights = coords[:, 3] - coords[:, 1]
    ycenter = (coords[:, 1] + coords[:, 3])/2
    xcenter = (coords[:, 0] + coords[:, 2])/2

    # Avoid NaN warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if window > 0:
            heights = generic_filter(heights, np.nanmedian, window)
            widths = generic_filter(widths, np.nanmedian, window)
            xcenter = generic_filter(xcenter, np.nanmedian, window)
            ycenter = generic_filter(ycenter, np.nanmedian, window)
            scores = generic_filter(scores, np.nanmedian, window)

    if dilation > 0:
        widths = widths * (1 + dilation)
        heights = heights* (1 + dilation)
    
    out = coords + 0
    out[:, 0] = (xcenter - widths/2).clip(0, im_width)
    out[:, 2] = (xcenter + widths/2).clip(0, im_width)
    out[:, 1] = (ycenter - heights/2).clip(0, im_height)
    out[:, 3] = (ycenter + heights/2).clip(0, im_height)
    return out, scores

def get_hoa_boxes(video_id, side='LEFT', typ='hand', object_threshold=0.0, hand_threshold=0.0):
    detections = load_detections(os.path.join(FLAGS.hoa_dir, video_id.split("_")[0], video_id + ".pkl"))
    dds = load_detections(os.path.join(FLAGS.hoa_dir, video_id.split("_")[0], video_id + ".pkl"))

    video_info = pd.read_csv(os.path.join(METADATA_PATH, "EPIC_100_video_info.csv"))
    resolution = video_info[video_info.video_id == video_id].iloc[0].resolution
    width, height = [int(x) for x in resolution.split('x')]
    coord_order = ["left", "top", "right", "bottom"]

    hand_boxes = np.zeros((len(detections), 5), dtype=np.float32)
    object_boxes = np.zeros((len(detections), 5), dtype=np.float32)
    
    hand_scores = -np.ones((len(detections), 1), dtype=np.float32)
    object_scores = -np.ones((len(detections), 1), dtype=np.float32)
    
    for i, d in enumerate(detections):
        # find highest scoring left hand, and take the box that is attached to it
        max_score = FLAGS.maxhandscore
        max_hand = None
        max_hand_ind = None
        for ind, h in enumerate(d.hands):
            if h.side.name == side and h.score > max_score:
                max_hand = h
                max_hand_ind = ind
                max_score = h.score
        
        hand_boxes[i,0] = i
        object_boxes[i,0] = i

        if max_hand is not None:
            hand_box = [getattr(max_hand.bbox, attr) for attr in coord_order]
            hand_boxes[i,1:] = hand_box
            hand_boxes[i,0] = i
            hand_scores[i,0] = max_score
            
            if len(d.objects) > 0:
                dd = dds[i]
                dd.scale(width_factor=width, height_factor=height)
                hand_object_idx_correspondences = dd.get_hand_object_interactions(
                    object_threshold=object_threshold, hand_threshold=hand_threshold)

                if max_hand_ind in hand_object_idx_correspondences.keys():
                    object_ind = hand_object_idx_correspondences[max_hand_ind]
                    object_box = [getattr(d.objects[object_ind].bbox, attr) 
                                  for attr in coord_order]
                    object_boxes[i,1:] = object_box
                    object_boxes[i,0] = i
                    object_scores[i,0] = d.objects[object_ind].score
    if typ == 'hand':
        boxes, scores = hand_boxes, hand_scores
    else:
        boxes, scores = object_boxes, object_scores
    return boxes, scores

def crop_out_tracks(video_id, boxes):
    crops = []
    for _, box in enumerate(boxes):
        i = int(box[0])
        box = box[1:]
        imgpath = f"{FLAGS.data_dir}/{video_id.split('_')[0]}/rgb_frames/{video_id}/frame_{str(i + 1).zfill(10)}.jpg"
        image = Image.open(imgpath)
        bbox_coords = [crd * image.size[j % 2] for j, crd in enumerate(box)]
        _bbox_coords = np.array([bbox_coords])
        bbox_coords = make_square(_bbox_coords,
            im_width=image.width, im_height=image.height)
        bbox_coords = bbox_coords[0].tolist()
        crop = image.crop(bbox_coords).resize((imsize, imsize))
        crops.append(crop)
    return crops

def split_into_tracks(boxes, scores):
    valid = scores[:,0] > -1
    starts = np.where(np.logical_and(valid[1:], np.invert(valid[:-1])))[0] + 1
    starts = starts.tolist()
    if valid[0]:
        starts.insert(0, 0)
    ends = np.where(np.logical_and(valid[:-1], np.invert(valid[1:])))[0] + 1
    ends = ends.tolist()
    if valid[-1]:
        ends.append(len(valid)-1)
    starts = np.array(starts)
    ends = np.array(ends)
    spans = np.array([starts, ends]).T
    
    tracks = []
    for i, j in spans:
        tracks.append(boxes[i:j,:])
    
    return spans, tracks

def generate_tracks(side, typ, out_dir, video_id):
    boxes, scores = get_hoa_boxes(video_id, side=side, typ=typ, object_threshold=0., hand_threshold=0.)
    fps_info = pd.read_csv(os.path.join(METADATA_PATH, 'EPIC_100_video_info.csv'))
    fps = fps_info[fps_info.video_id == video_id].iloc[0].fps
    sample_rate = int(fps / FLAGS.fps)
    
    boxes[scores[:,0] < 0, 1:] = np.nan
    scores[scores[:,0] < 0, :] = np.nan
    boxes[:,1:], scores[:,0] = filter_and_dilate_track(boxes[:,1:], scores[:,0],  
        window=FLAGS.median_filter_size*sample_rate, dilation=0.5)
    scores[np.isnan(scores)] = -1
    
    name = side[0].lower()
    spans, tracks = split_into_tracks(boxes, scores)
    track_out_dir = Path(out_dir) / typ
    track_out_dir.mkdir(parents=True, exist_ok=True)
    
    lens = spans[:,1] - spans[:,0]
    lens = lens / fps

    for ind, track in enumerate(tracks):
        if track.shape[0] > 20:
            crops = crop_out_tracks(video_id, track[::sample_rate,:])
            file_name = track_out_dir / f'{video_id}_{spans[ind, 0]}_{spans[ind, 1]}_{name}.jpg'
            save_trajectory(crops, file_name)

    return lens

def main(_):
    FLAGS.out_dir = f"{FLAGS.out_dir}_{FLAGS.maxhandscore}_{imsize}" 

    for split in ["train", "validation"]:
        annotations = pd.read_csv(
            os.path.join(METADATA_PATH, f"EPIC_100_{split}.csv"))
        
        # iterate on videos to load detection file once for all segments
        unique_videos = annotations.video_id.unique()
        unique_videos.sort()

        os.makedirs(FLAGS.out_dir, exist_ok=True)

        # Copy meta information to the target folder
        os.system(f"cp ../annotations/EPIC_55_annotations.csv {FLAGS.out_dir}/")
        os.system(f"cp ../annotations/EPIC_test_s1_object_video_list.csv {FLAGS.out_dir}/")
        os.system(f"cp ../annotations/EPIC_test_s2_object_video_list.csv {FLAGS.out_dir}/")

        out_dir = os.path.join(FLAGS.out_dir, split)

        print(len(unique_videos))
        # Filter videos (adapted from Rishabh)
        if FLAGS.filter:
            for i in list(range(26, 50)):
                unique_videos = [x for x in unique_videos if f'P{i}' not in x]
            unique_videos = [x for x in unique_videos if f'P02_130' not in x]
        
        print(len(unique_videos))
        unique_videos.sort()
        video_ids = unique_videos

        # Obtain and save tracks
        for typ in ['hand']:
            for side in ['LEFT', 'RIGHT']:
                pool = multiprocessing.Pool(processes=20)
                func = partial(generate_tracks, side, typ, out_dir)
                lengths_list = [l for l in tqdm(pool.imap_unordered(func, video_ids), total=len(video_ids))]
                lengths = np.concatenate(lengths_list, axis=0)
                if FLAGS.plot_hist:
                    fig, ax = subplot(plt, (1, 1))
                    bins = np.linspace(0, 30, 61)
                    ax.hist(lengths, bins=bins, width=0.9)
                    ax.set_xticks(bins[::2])
                    fig.savefig(str(Path(out_dir)) + f'/Hist_{side}_{typ}.pdf', bbox_inches='tight')
                    plt.close()

if __name__ == '__main__':
    app.run(main)

import os
import sys
import PIL
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import median_filter
from multiprocessing.pool import Pool

from epic_kitchens.hoa import load_detections
from epic_kitchens.hoa.types import HandSide

from tracking import get_tracks, MIN_TRACK_LENGTH

sys.path.append(str(Path(__file__).absolute().parent.parent))
from paths import *
from data.data_utils import get_splits


def load_image(path):
    try:
        frame_image = PIL.Image.open(path)
    except FileNotFoundError:
        frame_image = None
    return frame_image


def filter_and_dilate_track(track, window=10, dilation=0.2):
    """
    track : List[List] ordered [left, top, right, bottom]
    window: window size for median filtering
    returns List[List] ordered [left, top, right, bottom]
    """
    coords = np.array(track)
    widths = median_filter(coords[:, 2] - coords[:, 0], window) * (1 + dilation) * 228
    heights = median_filter(coords[:, 3] - coords[:, 1], window) * (1 + dilation) * 128
    side = np.maximum(heights, widths)
    xcenter = (coords[:, 1] + coords[:, 3]) / 2
    ycenter = (coords[:, 0] + coords[:, 2]) / 2
    coords[:, 0] = (ycenter - side / 456).clip(0, 1)
    coords[:, 2] = (ycenter + side / 456).clip(0, 1)
    coords[:, 1] = (xcenter - side / 256).clip(0, 1)
    coords[:, 3] = (xcenter + side / 256).clip(0, 1)
    return coords.tolist()


def extract_action_localized_tracks(split):
    """
    extract tracks looking at just bounding box coords
    does not use visual cues
    """

    output_path = DIR_TRACKS / "ioumf" / f"EPIC_100_{split}_tracks_ioumf.pkl"
    if output_path.exists():
        print("Track info already cached!")
        return False

    annotations = pd.read_csv(f"{DIR_ANNOTATIONS}/EPIC_100_train.csv")
    annotations.append(pd.read_csv(f"{DIR_ANNOTATIONS}/EPIC_100_validation.csv"), ignore_index=True)
    participants_in_split = get_splits(split)
    annotations = annotations[annotations.participant_id.isin(participants_in_split)]
    assert len(annotations.participant_id.unique()) == len(participants_in_split)
    
    # filter all actions less than MIN_TRACK_LENGTH
    annotations = annotations[(annotations.stop_frame - annotations.start_frame) > MIN_TRACK_LENGTH]

    # iterate on videos to load detection file once for all segments
    unique_videos = annotations.video_id.unique()
    T = []
    for video_id in tqdm(unique_videos, desc='Track Calculation'):
        # load detections
        video_detections = load_detections(DIR_SHAN_DETECTIONS / video_id[:3] / (video_id + '.pkl'))

        # extract tracks with temporal dilation
        tracks = get_tracks(video_id, video_detections, 0, None)

        # aggregate track info
        for track in tracks:
            T.append({"narration_id": video_id + "_01",
                    "track_id": video_id + "_01" + "_" + str(track["id"]),
                    **track})
    df = pd.DataFrame.from_dict(T)
    df.to_pickle(output_path)
    return True


def save_jpeg_for_vid(fps_info, all_tracks, split, gen_ioumf, gen_hands, vID, imsize=128, max_track_length=256, filter=True):
    coord_order = ["left", "top", "right", "bottom"]
    
    # load detected objects for each video
    video_detections = load_detections(Path(DIR_SHAN_DETECTIONS) / vID[:3] / f"{vID}.pkl")
    # get fps for frame sampling from video
    sample_rate = 5 if fps_info[fps_info.video_id == vID].iloc[0].fps == 50 else 6
    # filter out tracks for vID
    video_tracks = all_tracks[all_tracks.narration_id.str.contains(vID + "_")]

    metadata = {}
    for i, track in video_tracks.iterrows():
        ltt = len(track.track)
        # sample frames
        sampled_inds = [i for i in range(0, ltt, sample_rate)]
        tracked_boxids = [track.track[k] for k in sampled_inds]

        # load bboxes for video segments and optionally smooth size, then sample
        tracked_bboxes_all = [[getattr(video_detections[fr].objects[ob].bbox, attr) for attr in coord_order] for fr, ob in track.track]

        # Store hand object mapping. When two hands interact with the same object, pick the hand which has been picked the most previously.
        obj_hand_maps = []
        handside_sums = {HandSide.LEFT:0, HandSide.RIGHT:0}
        for i in range(len(track.track)):
            fr, obj_id = track.track[i]
            hand_dets = video_detections[fr].hands
            raw_hand_obj_map = {k:v for k,v in video_detections[fr].get_hand_object_interactions(0.1, 0.1).items() if v==obj_id}
            left_scores = [h.score for h in hand_dets if h.side == HandSide.LEFT]
            right_scores = [h.score for h in hand_dets if h.side == HandSide.RIGHT]
            left_max = np.max(left_scores) if len(left_scores) else 0
            right_max = np.max(right_scores) if len(right_scores) else 0
            seen = set()
            hand_obj_map = {}
            for hand, obj in raw_hand_obj_map.items():
                max_score = left_max if hand_dets[hand].side == HandSide.LEFT else right_max 
                if hand_dets[hand].side not in seen and hand_dets[hand].score >= max_score:
                    hand_obj_map[hand] = obj
                    seen.add(hand_dets[hand].side)
            assert len(hand_obj_map) <= 2, hand_dets
            oh_map = {}
            if len(hand_obj_map) == 0:
                obj_hand_maps.append(oh_map)
                continue
            for hand in hand_obj_map:
                handside_sums[video_detections[fr].hands[hand].side] += 1
            if len(hand_obj_map) == 1:
                oh_map[obj_id] = list(hand_obj_map.keys())[0]
            else:
                best_hand_side = HandSide.LEFT if handside_sums[HandSide.LEFT] > handside_sums[HandSide.RIGHT] else HandSide.RIGHT
                oh_map[obj_id] = 0 if video_detections[fr].hands[0].side == best_hand_side else 1
            obj_hand_maps.append(oh_map)
        
        tracked_bboxes_hands = [[getattr(video_detections[fr].hands[obj_hand_maps[i][ob]].bbox, attr) for attr in coord_order] if ob in obj_hand_maps[i] else None for i, (fr, ob) in enumerate(track.track)]

        track_metadata = {"frames":[], "objects":[], "hands":[]}
        if filter:
            tracked_bboxes_all = filter_and_dilate_track(tracked_bboxes_all)
            valid_hand_bboxes = [b for b in tracked_bboxes_hands if b is not None]
            if len(valid_hand_bboxes) > 0:
                valid_hand_bboxes = filter_and_dilate_track(valid_hand_bboxes)
            count = 0
            for i in range(len(tracked_bboxes_hands)):
                if tracked_bboxes_hands[i] is not None:
                    tracked_bboxes_hands[i] = valid_hand_bboxes[count]
                    count += 1
        tracked_bboxes = [tracked_bboxes_all[k] for k in sampled_inds]
        tracked_bboxes_hands = [tracked_bboxes_hands[k] for k in sampled_inds]

        track_metadata["frames"] = [fr for fr, _ in tracked_boxids]
        track_metadata["hands"] = [video_detections[fr].hands[obj_hand_maps[i][ob]] if ob in obj_hand_maps[i] else None for i, (fr, ob) in enumerate(track.track)]
        track_metadata["hands"] = [track_metadata["hands"][k] for k in sampled_inds]
        track_metadata["objects"] = [video_detections[fr].objects[ob] for fr, ob in tracked_boxids]
        
        # iterate over bboxes in sampled track and extract segment of image
        detected_object_track = []
        detected_hand_track = []
        for xx, (fr, ob) in enumerate(tracked_boxids):
            imgpath = f"{DIR_RGB_FRAMES}/{vID[:3]}/rgb_frames/{vID}/frame_{str(fr + 1).zfill(10)}.jpg"
            frame_image = load_image(imgpath)
            assert frame_image is not None, f'Failed to load {imgpath}.' 

            if gen_ioumf:
                bbox_coords = [crd * frame_image.size[j % 2] for j, crd in enumerate(tracked_bboxes[xx])]
                crop = frame_image.crop(bbox_coords).resize((imsize, imsize), PIL.Image.ANTIALIAS)
                detected_object_track.append(crop)
            else:
                detected_object_track.append(None)

            if gen_hands and tracked_bboxes_hands[xx] is not None:
                bbox_coords = [crd * frame_image.size[j % 2] for j, crd in enumerate(tracked_bboxes_hands[xx])]
                crop = frame_image.crop(bbox_coords).resize((imsize, imsize), PIL.Image.ANTIALIAS)
                detected_hand_track.append(crop)
            else:
                detected_hand_track.append(None)

            # track_metadata["hand_valid_indices"].append(tracked_bboxes_hands[xx] is not None)
            # track_metadata["frames"].append(fr)

        # split track if too long
        for s, seg in enumerate(range(0, len(detected_object_track), max_track_length)):
            segmented_track = detected_object_track[seg:seg+max_track_length]
            segmented_track_hand = detected_hand_track[seg:seg+max_track_length]
            if len(segmented_track) < MIN_TRACK_LENGTH//sample_rate:
                break
            
            # seg_dir = base_dir / "tracks" / f"{split}_ioumf"                
            # seg_dir_hand = base_dir / "tracks_hand" / f"{split}_ioumf"
            seg_dir = Path(DIR_TRACKS) / "ioumf" / "images"                
            seg_dir_hand = Path(DIR_TRACKS_HAND) / "ioumf" / "images"
            seg_dir.mkdir(parents=True, exist_ok=True)
            seg_dir_hand.mkdir(parents=True, exist_ok=True)
            track_name = f"{track.track_id}_{s}.jpg"
            metadata[track_name] = {k:v[seg:seg+max_track_length] for k,v in track_metadata.items()}
            metadata[track_name]["length"] = len(segmented_track)
            metadata[track_name]["pid"] = vID[:3]
            
            if gen_ioumf:
                new_im = PIL.Image.new('RGB', (imsize * len(segmented_track), imsize))
            if gen_hands:
                new_im_hand = PIL.Image.new('RGB', (imsize * len(segmented_track), imsize))
            for imnum, im in enumerate(segmented_track):
                if gen_ioumf:
                    new_im.paste(im, (imnum * imsize, 0))
                if gen_hands and segmented_track_hand[imnum] is not None:
                    new_im_hand.paste(segmented_track_hand[imnum], (imnum * imsize, 0))
            if gen_ioumf:
                new_im.save(seg_dir / track_name)
            if gen_hands:
                new_im_hand.save(seg_dir_hand / track_name)


    metadata = pd.DataFrame.from_dict(metadata, orient='index')
    metadata.index.name = 'track_name'
    return metadata

            

def save_tracks_as_jpeg(imsize=128, max_track_length=256, filter=True, split="train", num_workers=0, gen_ioumf=True, gen_hands=False):
    # load tracks and extract unique video ids
    tracks_path = f"{DIR_TRACKS}/ioumf/EPIC_100_{split}_tracks_ioumf.pkl"
    all_tracks = pd.read_pickle(tracks_path)
    video_ids = ["_".join(nID.split("_")[:2]) for nID in all_tracks.narration_id.unique()]

    # load sampling info for frames
    fps_info = pd.read_csv(f"{DIR_ANNOTATIONS}/EPIC_100_video_info.csv")

    # iterate over all tracks in each video
    wrapped_func = functools.partial(save_jpeg_for_vid, fps_info, all_tracks, split, gen_ioumf, gen_hands)
    if num_workers == 0:
        tq = tqdm(video_ids)
        metadata = []
        for vID in tq:
            tq.set_description(vID)
            m = wrapped_func(vID)
            metadata.append(m)
    else:
        with Pool(num_workers) as p:
            metadata = list(tqdm(p.imap(wrapped_func, video_ids), total=len(video_ids), desc='Track Saving', dynamic_ncols=True, leave=True))
    
    metadata = pd.concat(metadata)
    metadata["split"] = split
    return metadata
    

if __name__ == "__main__":
    md_path = DIR_TRACKS / "ioumf"
    md_path.mkdir(exist_ok=True, parents=True)
    dfs = []
    splits = ['train', 'validation']
    for split in splits:
        extract_action_localized_tracks(split)
        df = save_tracks_as_jpeg(split=split, num_workers=64, gen_ioumf=True, gen_hands=True)
        df['hand_valid_indices'] = [np.where(np.array(h) != None)[0] for h in df.hands]
        df.to_pickle(md_path / f"track_detections_{split}.pkl")
        dfs.append(df)

import os, sys
import argparse
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.patches as patches
import numpy as np
import multiprocessing
from functools import partial

Image.MAX_IMAGE_PIXELS = None
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import img_as_ubyte
import pickle

MAX_TRACK_LEN = 256


def get_bbox(bb, npimg):
    st = bb
    for char in ['[', ']', '(', ')']:
        st = st.replace(char, '')
    bbox = [int(s) for s in st.split(',')]
    # print(npimg.shape)

    y = int(bbox[0] * npimg.shape[0] / 1080)
    x = int(bbox[1] * npimg.shape[1] / 1920)
    h = int(bbox[2] * npimg.shape[0] / 1080)
    w = int(bbox[3] * npimg.shape[1] / 1920)

    midx = x + w // 2
    midy = y + h // 2
    sz = max(w, h)
    sz = int(1.2 * sz)
    x = max(0, midx - sz // 2)
    y = max(0, midy - sz // 2)
    right = min(npimg.shape[1] - 1, midx + sz // 2)
    bottom = min(npimg.shape[0] - 1, midy + sz // 2)

    return x, y, right, bottom


def crop_resize(img, tup, shape=(128, 128)):
    cropped_img = img.crop(tup)
    resized_img = cropped_img.resize(shape, Image.ANTIALIAS)

    return resized_img


def convex_sum(tup1, tup2, ind, length):
    lam = 1 - (ind / length)

    return (int(lam * i1 + (1 - lam) * i2) for i1, i2 in zip(tup1, tup2))


def get_path(data_path, row, frame):
    return os.path.join(data_path, row['participant_id'], "rgb_frames", row['video_id'],
                        f"frame_{format(frame, '010d')}.jpg")


def save_image(annot, action, data_path, split, obj):
    annot_obj = annot[annot['noun'] == obj]
    action_obj = action[action['noun'] == obj]

    if len(annot_obj) == 0:
        return

    obj_name = annot_obj['noun'].iloc[0]
    os.makedirs(f"/data01/rgoyal6/datasets/EPIC-KITCHENS/tracks/{split}_cleanfree/", exist_ok=True)

    for vid in list(set(list(annot_obj['video_id']))):
        annot_vid = annot_obj[annot_obj['video_id'] == vid]
        if len(annot_vid) == 0:
            continue
        sorted_frames = list(annot_vid.sort_values("frame").frame)
        seg_starts = [sorted_frames[0]]
        seg_stops = []
        for i in range(len(sorted_frames) - 1):
            if sorted_frames[i + 1] - sorted_frames[i] == 30:
                continue
            seg_stops.append(sorted_frames[i])
            seg_starts.append(sorted_frames[i + 1])
        seg_stops.append(sorted_frames[-1])
        assert len(seg_starts) == len(seg_stops)
        #################
        for i in range(len(seg_starts)):
            # row = action_vid.iloc[i]
            closest_start = seg_starts[i]
            # annot_vid.iloc[(annot_vid['rgb_frame']-row['start_frame']).abs().argsort()[0:1]]
            closest_stop = seg_stops[i]
            # annot_vid.iloc[(annot_vid['rgb_frame']-row['stop_frame']).abs().argsort()[0:1]]
            # print(closest_start, closest_stop, row)
            # print(closest_start['rgb_frame'].iloc[0])
            if os.path.isdir(f"{data_path}/{vid}_{obj.replace(' ', '')}"):
                path_list = [
                    f"{data_path}/{vid}_{obj.replace(' ', '')}/{vid}_{format(i, '010d')}.jpg" \
                    for i in range(closest_start, closest_stop + 1)]
                if len(path_list) > 5:
                    image = create_image_list(path_list)
                    # image = [img for i, img in enumerate(image) if i % 2 == 0]
                    if image is not None:
                        for j in range(0, np.ceil(image.shape[1] // image.shape[0] / MAX_TRACK_LEN).astype(np.int32)):
                            cut_img = image[:,
                                      j * (image.shape[0] * MAX_TRACK_LEN): (j + 1) * (image.shape[0] * MAX_TRACK_LEN)]
                            if cut_img.shape[1] // image.shape[0] >= 4:
                                skimage.io.imsave(
                                    f"/data01/rgoyal6/datasets/EPIC-KITCHENS/tracks/{split}_cleanfree/{vid}_{obj}_{i}_{j}.jpg",
                                    img_as_ubyte(cut_img))

    print(f"Finished {obj_name} {len(annot_obj)}")


def create_image_list(lst):
    images = []
    count = 0
    for path in lst:
        if os.path.isfile(path):
            count += 1
            if count % 2:
                img = skimage.io.imread(path)
                images.append(img)
    if len(images) == 0:
        return None
    return np.concatenate(images, axis=1)


def filter_data(data, filt):
    filtered_ids = [data[key].isin(val).values for key, val in filt.items()]
    filtered_ids = filtered_ids[0] if len(filtered_ids) == 1 else np.logical_and(*filtered_ids)

    data_filtered = data[filtered_ids]

    return data_filtered


def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            return pickle.load(file)
        except EOFError:
            pass


def main(args):
    data_path = os.path.join(args.data)
    annot_path = args.annot
    corres_path = args.corres
    action_path = args.action

    filters = read_from_pickle(args.filters)

    annot = pd.read_csv(annot_path)
    annot = filter_data(annot, filters[args.split])

    # Remove empty bounding box frames
    ind = annot['bounding_boxes'].str.len() > 2
    annot = annot[ind]

    class2obj = {i: j for i, j in zip(list(annot['noun']), list(annot['noun_class']))}

    corres = pd.read_csv(corres_path)

    annot['narr_id'] = [f"{i}_{j}" for i, j in zip(list(annot['video_id']), list(annot['frame']))]
    corres['narr_id'] = [f"{i}_{j}" for i, j in zip(list(corres['video_id']), list(corres['object_frame']))]

    dic = {i: j for i, j in zip(list(corres['narr_id']), list(corres['action_frame']))}
    lt = [dic[i] for i in list(annot['narr_id'])]

    annot['rgb_frame'] = lt

    action = pd.read_csv(action_path)
    action = filter_data(action, filters[args.split])
    action['narr_id_start'] = [f"{i}_{j}" for i, j in zip(list(action['video_id']), list(action['start_frame']))]
    action['narr_id_stop'] = [f"{i}_{j}" for i, j in zip(list(action['video_id']), list(action['stop_frame']))]

    # annot = annot[annot['video_id'].isin([f"P01_{format(i, '02d')}" for i in range(1,2)])]
    # annot = annot[annot['participant_id'].isin([f"P{format(i, '02d')}" for i in range(2, 3)])]
    # action = action[action['participant_id'].isin([f"P{format(i, '02d')}" for i in range(2, 3)])]
    # sys.exit()

    # object_list = ['bin', 'plate', 'fridge',
    # 				 'spoon', 'knife', 'tap',
    # 				 'oven', 'microwave', 'jar',
    # 				 'bottle', 'bottles', 'egg',
    # 				 'onion', 'eggs', 'flour',
    # 				 'microwave', 'drawer'
    # 				 ]
    object_list = list(set(list(annot['noun'])))
    # print(object_list)
    class_list = [class2obj[i] for i in object_list]
    # print(len(class_list) - len(np.unique(class_list)))
    # sys.exit()

    pool = multiprocessing.Pool(processes=6)

    func = partial(save_image, annot, action, data_path, args.split)
    pool.map(func, object_list)


# print("Finished creating Dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training hyper-parameters')

    # parser.add_argument('--data', dest='data', type=str,
    # 					default="/data01/rgoyal6/datasets/EPIC-KITCHENS/")
    parser.add_argument('--data', dest='data', type=str,
                        default="/home/mohit/VOS/EPIC-2018-frame/dataset")
    parser.add_argument('--annot', dest='annot', type=str,
                        default="/home/mohit/VOS/EPIC-clean-clips/EPIC_train_object_labels.csv")
    parser.add_argument('--corres', dest='corres', type=str,
                        default="/home/mohit/VOS/EPIC-clean-clips/EPIC_train_object_action_correspondence.csv")
    parser.add_argument('--action', dest='action', type=str,
                        default="/home/mohit/VOS/EPIC-clean-clips/EPIC_100_train.csv")
    parser.add_argument('--filters', dest='filters', type=str, default="/home/mohit/VOS/EPIC-2018/filters.pkl")
    parser.add_argument('--split', dest='split', type=str, default="testunseen")

    # parser.add_argument('--config', dest='config', type=str,
    #                     default="./src/configs/base.yaml")
    # parser.add_argument('--data', dest='data', type=str,
    #                     default="/home/mohit/VOS/AnnotatedData/")
    # parser.add_argument('--model', dest='model', type=str,
    #                     default="./models/EPIC-KITCHENS/")
    # parser.add_argument('--log', dest='log', type=str,
    #                     default="./logs/EPIC-KITCHENS/")
    # parser.add_argument('-name', dest='name', type=str, default="timecontrastnofilterrandom_fridge")

    # parser.add_argument('--config', dest='config', type=str,
    #                     default="./src/configs/base.yaml")
    # parser.add_argument('--data', dest='data', type=str,
    #                     default="/data01/mohit/PennAction/processedcrop/")
    # parser.add_argument('--model', dest='model', type=str,
    #                     default="./models/PennAction/")
    # parser.add_argument('--log', dest='log', type=str,
    #                     default="./logs/PennAction/")
    # parser.add_argument('-name', dest='name', type=str, default="timecontrast")

    args = parser.parse_args()
    main(args)

from enum import unique
import sys
import pdb
from pathlib import Path
from PIL import Image, ImageEnhance, ImageStat
import argparse
from matplotlib import colors
from tqdm import tqdm
import cv2
import torch
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# sys.path.append(str(Path(__file__).parent.parent.absolute()))
# from model import Resnet, SimCLRLayer, SimCLRLayerMultiHead

dic_opp = {
    'open': ['close'],
    'close': ['open'],
    'inhand': ['outofhand'],
    'outofhand': ['inhand'],
    'peeled': ['unpeeled'],
    'unpeeled': ['peeled'],
    'whole': ['cut'],
    'cut': ['whole'],
    'cooked': ['raw'],
    'raw': ['cooked'],
}
classes = list(dic_opp.keys())
classes.sort()

colors_per_class = None

def fix_random_seeds():
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image

def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, indices, labels, plot_size=1500, max_image_size=64):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, idx, label, x, y in tqdm(
            zip(images, indices, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        if idx < 0:
            image = Image.open(image_path)
        else:
            image = Image.open(image_path).crop((128*idx, 0, (idx+1)*128, 128))
        
        enhancer = ImageEnhance.Contrast(image)
        stat = ImageStat.Stat(image)
        r,g,b = stat.mean
        brightness = np.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
        if brightness < 50:
            image = enhancer.enhance(factor=3.0)
        
        image = np.asarray(image)[:,:,::-1]

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    img = Image.fromarray(tsne_plot[:, :, ::-1])
    padding = 100
    img_padded = Image.new(img.mode, (img.size[0] + padding, img.size[1] + padding), (255, 255, 255))
    img_padded.paste(img, (padding//2, padding//2))
    img_padded.save("tsne_images.png")
    


def visualize_tsne_points(tx, ty, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.savefig("tsne_points.png")


def visualize_tsne(tsne, images, indices, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # visualize the plot: samples as images
    visualize_tsne_images(tx, ty, images, indices, labels, max_image_size=max_image_size)


def get_features(predictions_path, num_tsne_datapoints=4000):
    global colors_per_class
    predictions = torch.load(predictions_path)

    sample_i = np.random.permutation(predictions['embedding'].shape[0])[:num_tsne_datapoints]
    features = predictions["embedding"][sample_i]
    image_paths = predictions["path"][sample_i]
    indices = predictions["idx"][sample_i]
    # pdb.set_trace()

    df = pd.read_csv("/home/smodi9/epic_kitchens/evaluationVOS/Annotations/nov05_2021_fullbatch.csv")
    df.set_index("path", inplace=True)
    dict_labels = df["state"].to_dict()
    labels = []
    for p in image_paths:
        raw_label = dict_labels[p].split(",")
        label = ""
        for cls in classes:
            if cls in raw_label:
                label += cls + ","
        labels.append(label[:-1])
    labels = np.array(labels)

    # labels = np.array(["default"]*num_tsne_datapoints)

    return features, image_paths, indices, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', '-p', type=Path, required=True)
    args = parser.parse_args()

    fix_random_seeds()

    features, image_paths, indices, labels = get_features(args.predictions)

    tsne = TSNE(n_components=2).fit_transform(features)

    unique_labels = list(sorted(set(labels)))
    df = pd.DataFrame.from_dict({"labels":labels})
    # pdb.set_trace()
    counts = df.labels.value_counts()
    valid_labels = counts[:10].index.values
    valid_i = np.where(df.labels.isin(valid_labels))[0]
    sample_i = valid_i[:200]


    # sample_i = np.random.permutation(tsne.shape[0])[:200]
    tsne = tsne[sample_i]
    image_paths = image_paths[sample_i]
    indices = indices[sample_i]
    labels = labels[sample_i]

    unique_labels = list(sorted(set(labels)))
    colrs = list(cm.jet(np.linspace(0, 1, len(unique_labels))))

    global colors_per_class
    colors_per_class = {lab:(colrs[i][:3]*255).astype(int).tolist() for i,lab in enumerate(unique_labels)}
    print(len(unique_labels))
    print(colors_per_class)

    visualize_tsne(tsne, image_paths, indices, labels)

if __name__ == '__main__':
    main()
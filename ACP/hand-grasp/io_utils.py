import numpy as np
import _pickle as cPickle, os, time
from PIL import Image

def save_trajectory(images, impath):
    width = images[0].size[0]
    height = images[0].size[1]
    rows = cols = int(np.ceil(np.sqrt(len(images))))
    new_im = Image.new(images[0].mode, (width*cols, height*rows))
    for imnum, im in enumerate(images):
        i, j = imnum % cols, imnum // cols
        new_im.paste(im, (i*width, j*height))
    ee = new_im.getexif()
    ee[0] = len(images)
    ee[1] = width
    ee[2] = height
    return new_im.save(impath, exif=ee)

def stack_images(images, horizontal=True):
    width = images[0].size[0]
    height = images[0].size[1]
    if horizontal:
        rows = 1
        cols = len(images)
    else:
        rows = len(images)
        cols = 1

    new_im = Image.new(images[0].mode, (width*cols, height*rows))
    for imnum, im in enumerate(images):
        i, j = imnum % cols, imnum // cols
        new_im.paste(im, (i*width, j*height))
    
    return new_im

def load_trajectory(impath):
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

def get_trajectory_atindices(images, indices):
    exif = images.getexif()
    num_images = exif[0]
    width = exif[1]
    height = exif[2]
    cols = images.size[0] // exif[1]
    rows = images.size[1] // exif[2]
    all_images = []
    for imnum in indices:
        i, j = imnum % cols, imnum // cols
        image = images.crop((i*width, j*height, i*width + width , j*height + height))
        all_images.append(image)
    return all_images

def get_trajectory_atindices_fromfolder(path, indices):
    all_images = []
    for imnum in indices:
        impath = os.path.join(path, f"{imnum}.jpg")
        image = Image.open(impath)
        all_images.append(image)
    return all_images

def get_trajectory_with_length(impath):
    images = Image.open(impath)
    exif = images.getexif()
    num_images = exif[0]

    return num_images, images

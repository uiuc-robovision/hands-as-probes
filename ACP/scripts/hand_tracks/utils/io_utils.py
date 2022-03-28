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

def save_variables(pickle_file_name, var, info, overwrite=False):
  if os.path.exists(pickle_file_name) and overwrite == False:
    raise Exception('{:s} exists and over write is false.'.format(pickle_file_name))
  # Construct the dictionary
  assert(type(var) == list); assert(type(info) == list);
  for t in info: assert(type(t) == str), 'variable names are not strings'
  d = {}
  for i in xrange(len(var)):
    d[info[i]] = var[i]
  with open(pickle_file_name, 'w') as f:
    cPickle.dump(d, f, cPickle.HIGHEST_PROTOCOL)

def load_variables(pickle_file_name):
  if os.path.exists(pickle_file_name):
    with open(pickle_file_name, 'r') as f:
      d = cPickle.load(f)
    return d
  else:
    raise Exception('{:s} does not exists.'.format(pickle_file_name))

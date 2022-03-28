import sys
import yaml
import logging
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, silhouette_score
import matplotlib
import matplotlib.font_manager
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import torch, random

matplotlib.font_manager._rebuild()
plt.rc('font',family='Times New Roman')

import pickle

def set_torch_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

def load_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class Logger:
    def __init__(self, logger_name='__name__', file_name='log/logs.log'):
        # Create a custom logger
        self.logger = logging.getLogger(logger_name)

        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()

        self.logger.setLevel(logging.DEBUG)
        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.DEBUG)
        f_handler = logging.FileHandler(file_name, mode='w')
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def close(self):
        logging.shutdown()


class Meter:

    def __init__(self):
        self.val = 0
        self.count = 0
        self.lowest = float("inf")
        self.highest = float("-inf")
        self.avg_list = []

    def update(self, val, count=1):
        self.val += val
        self.count += count

    def reset(self):
        self.val = 0
        self.count = 0

    def check(self, highest=False):
        avg = self.average()
        print(f"Avg {avg} Highest {self.highest} Lowest {self.lowest}")
        if highest:
            if avg > self.highest:
                self.highest = avg
                return 1
        else:
            if avg < self.lowest:
                self.lowest = avg
                return -1
        return 0

    def average(self):
        avg = self.val / (self.count + 1e-5)
        self.avg_list.append(avg)
        return avg


class APMeterBin:

    def __init__(self):
        self.probs = []
        self.labels = []
        self.rng = np.random.default_rng(0)

    def add(self, probs, labels):
        self.probs.append(probs)
        self.labels.append(labels)

    def value(self):
        if len(self.labels) > 0:
            labs = np.concatenate(self.labels, axis=0).astype(np.int32)
            probs = np.concatenate(self.probs, axis=0)

            precision = average_precision_score(labs, probs)

            return precision
        else:
            return 0

    def reset(self):
        self.probs = []
        self.labels = []

    def value_random(self, size=20, downsample=10):
        if len(self.labels) > 0:
            ind1 = self.rng.choice(np.arange(len(self.labels)), size=min(size, len(self.labels)), replace=False)
            labs = np.concatenate([self.labels[i][::downsample] for i in ind1], axis=0).astype(np.int32)
            probs = np.concatenate([self.probs[i][::downsample] for i in ind1], axis=0)

            # ind = self.rng.choice(np.arange(len(labs)), size=min(size, len(labs)), replace=False)
            # labs = labs[ind]
            # probs = probs[ind]

            precision = average_precision_score(labs, probs)

            return precision
        else:
            return 0

    def get_acc(self):
        labs = np.concatenate(self.labels, axis=0)
        probs = np.concatenate(self.probs, axis=0)
        acc = np.mean(((probs > 0.5)*1. == labs)*1.0)
        return acc

    def to_pickle(self, path):
        with open('{}'.format(path), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

class APMeterCustom:

    def __init__(self, classes):
        self.classes = classes
        self.probs = []
        self.labels = []

    def add(self, probs, labels):
        self.probs.append(probs)
        self.labels.append(labels)

    def value(self, use_opposite=True, return_mean=False):
        APs = {}
        labs = np.concatenate(self.labels, axis=0)
        probs = np.concatenate(self.probs, axis=0)

        for klass in range(len(self.classes)):
            positives = probs[labs == klass]
            negatives = probs[labs != klass]
            l_class = np.concatenate([np.ones((len(positives))), np.zeros((len(negatives)))], axis=0)
            p_class = np.concatenate([positives[:, klass], negatives[:, klass]], axis=0)
            if (len(positives) > 0 and len(negatives) > 0):
                precision = average_precision_score(l_class, p_class)
                APs[self.classes[klass]] = precision
            else:
                precision = 0

        if return_mean:
            return np.mean(list(APs.values()))
        return APs

    def get_acc(self):
        labs = np.concatenate(self.labels, axis=0)
        probs = np.concatenate(self.probs, axis=0)
        acc = np.mean((np.argmax(probs, axis=1) == labs)*1.)
        return acc

    def get_cm(self, normalize="true"):
        labs = np.concatenate(self.labels, axis=0)
        probs = np.concatenate(self.probs, axis=0)
        preds = np.argmax(probs, axis=1)

        cm = confusion_matrix(labs, preds, normalize=normalize)

        return cm

    def plot_cm(self, normalize="true", path=None):
        if path is not None:
            labs = np.concatenate(self.labels, axis=0)
            probs = np.concatenate(self.probs, axis=0)
            preds = np.argmax(probs, axis=1)
            cm = confusion_matrix(labs, preds, normalize=normalize)
            cm_ob = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.classes)
            factor = max(1, len(cm) // 15)
            fig = plt.figure(figsize=(10*factor,8*factor))
            cm_ob.plot(values_format='0.2f', ax=fig.gca())
            plt.tight_layout()
            plt.savefig(path, dpi=300)

    def to_pickle(self, path):
        with open('{}'.format(path), 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(fp):
	return pickle.load(open(fp, "rb"))

def create_imagewtext(size, text, fsize=30, RGB=True):
    if RGB:
        img = Image.new("RGB", size)
    else:
        img = Image.new("L", size)
    draw = ImageDraw.Draw(img)
    shape = img.size
    font = ImageFont.truetype("/home/mohit/fonts/fonts/SF Willamette.ttf", fsize)
    if RGB:
        draw.text((0, shape[1]//2 - fsize // 2), text, (255, 25, 255), font=font)
        return img
    
    draw.text((0, shape[1]//2 - fsize // 2), text, (255), font=font)
    return img

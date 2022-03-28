import sys
import yaml
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import argparse
from pathlib import Path
import pdb

global_logger = None

def load_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def init_logger(log_path):
    global global_logger
    assert global_logger is None, "Logger already initialized!"
    global_logger = Logger("main", log_path)
    return global_logger

# Create a custom logger
class Logger(logging.Logger):
    def __init__(self, logger_name='__name__', file_name='log/logs.log'):
        super(Logger, self).__init__(logger_name, logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.DEBUG)
        f_handler = logging.FileHandler(file_name)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.addHandler(c_handler)
        self.addHandler(f_handler)

        self.print_once_flags = set()

    def log_once(self, msg, *args):
        if msg not in self.print_once_flags:
            self.info(msg, *args)
            self.print_once_flags.add(msg)


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

    def check(self):
        if self.val > self.highest:
            self.highest = self.val
            return 1
        if self.val < self.lowest:
            self.lowest = self.val
            return -1
        return 0

    def average(self):
        avg = self.val / self.count
        self.avg_list.append(avg)
        return avg


class AccuracyMeter:
    def __init__(self):
        self.accs = []

    def update(self, logits, labels):
        accs = logits.max(1)[1] == labels
        # pdb.set_trace()
        self.accs.append(accs.detach().cpu().numpy())

    def average(self):
        accs = np.concatenate(self.accs)
        return accs.mean()

    def reset(self):
        self.accs.clear()


class ConfEvaluator(object):
    def __init__(self, n_classes, device):
        self.n_classes = n_classes
        self.device = device
        self.include = torch.arange(self.n_classes)
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes), device=self.device,
                                       dtype=torch.double, requires_grad=False)
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used

        # meter related stuff
        self.acc = 0
        self.highest = float("-inf")

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes), device=self.device, dtype=torch.double)
        self.ones = None
        self.last_scan_size = None

    def add_batch(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device, dtype=torch.double)
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix.index_put_(tuple(idxs), self.ones, accumulate=True)

    def get_stats(self):
        # remove fp and fn from confusion on the ignore classes cols and rows
        conf = self.conf_matrix.clone()

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def get_iou(self):
        tp, fp, fn = self.get_stats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou

    def get_precision_recall(self):
        tp, fp, fn = self.get_stats()
        precision = tp/(tp + fp + 1e-15)
        recall = tp/(tp + fn + 1e-15)
        return list(zip(precision.tolist(), recall.tolist()))

    def get_acc(self):
        tp, fp, fn = self.get_stats()
        total_tp = tp.sum()
        total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
        acc_mean = total_tp / total
        self.acc = acc_mean
        return acc_mean

    def check(self):
        if self.acc > self.highest:
            self.highest = self.acc
            return 1
        return 0

def save_plot(vals, labels, name):
    fig = plt.figure()
    for i, v in enumerate(vals):
        x = [j for j in range(len(v))]
        plt.plot(x, v, label=labels[i])
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(name)


def extract_vals(logpath, kws):
    vals = {k: [] for k in kws}
    with open(logpath, "r") as f:
        for line in f:
            for k in kws:
                idx = line.lower().find(k)
                if idx >= 0:
                    while line[idx] != ":":
                        idx += 1
                    vals[k].append(float(line[idx + 2: idx + 10]))
    return vals

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # worker_info = torch.utils.data.get_worker_info()
    # worker_id = worker_info.id
    # dataset = worker_info.dataset
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    # dataset.rng = np.random.default_rng((torch.initial_seed() + worker_id) % np.iinfo(np.int32).max)

def get_worker_generator(seed=0):
    g = torch.Generator()
    g.manual_seed(0)
    return g

def set_torch_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

def worker_init_fn_mohit(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    dataset = worker_info.dataset
    dataset.rng = np.random.default_rng((torch.initial_seed() + worker_id) % np.iinfo(np.int32).max)


def load_base_model(model_path, model, device):
    model_info = torch.load(model_path, map_location=device)
    model_weights = {k: model_info["model_state_dict"]["network." + k]
                        for k in model.network.state_dict().keys()}
    model.network.load_state_dict(model_weights)

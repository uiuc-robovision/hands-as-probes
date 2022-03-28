import sys
import pdb
import yaml
import torch
import logging
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def load_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


class Logger:
    def __init__(self, logger_name='__name__', file_name='log/logs.log', mode='a'):
        # Create a custom logger
        self.logger = logging.getLogger(logger_name)

        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()

        self.logger.setLevel(logging.DEBUG)
        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.DEBUG)
        f_handler = logging.FileHandler(file_name, mode=mode)
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
        self.last = float("-inf")
        self.avg_list = []

    def update(self, val, count=1):
        self.val += val
        self.count += count

    def reset(self):
        self.val = 0
        self.count = 0

    def check(self, highest=True):
        avg = self.average()
        self.last = avg
        # print(f"Avg {avg} Highest {self.highest} Lowest {self.lowest}")
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
        avg = self.val / self.count
        self.avg_list.append(avg)
        return avg


class APMeterCustom:
    def __init__(self, dic2num, classes):
        self.dic2num = dic2num
        self.classes = classes
        self.probs = []
        self.labels = []
        self.pr_curve = {}

    def add(self, probs, labels):
        self.probs.append(probs)
        self.labels.append(labels)

    def value(self, use_opposite=True, return_mean=False, weighted=False):
        APs = {}
        APs_lens = {}
        labs = np.concatenate(self.labels, axis=0)
        probs = np.concatenate(self.probs, axis=0)
        for klass in self.dic2num.keys():
            positives = probs[labs[:, klass] == 1]
            if use_opposite:
                if self.dic2num[klass] is not None:
                    negatives = probs[(labs[:, klass] == 0) & (np.sum(labs[:, self.dic2num[klass]], axis=1) > 0)]
                else:
                    negatives = probs[(labs[:, klass] == 0)]
            else:
                negatives = probs[(labs[:, klass] == 0)]
            l_class = np.concatenate([np.ones((len(positives))), np.zeros((len(negatives)))], axis=0)
            p_class = np.concatenate([positives[:, klass], negatives[:, klass]], axis=0)
            if (len(positives) > 0 and len(negatives) > 0):
                precision = average_precision_score(l_class, p_class)
                precisions, recalls, thresholds = precision_recall_curve(l_class, p_class)
                APs[self.classes[klass]] = precision
                APs_lens[self.classes[klass]] = p_class.shape[0]
                self.pr_curve[self.classes[klass]] = (precisions, recalls, thresholds)
            else:
                precision = 0

        if return_mean:
            scores = np.array(list(APs.values()))
            lens = np.array(list(APs_lens.values()))
            if weighted:
                return (scores * lens / lens.sum()).sum()
            return scores.mean()
        return APs, self.pr_curve

class APMeterSingleClassClassification:
    def __init__(self, classes):
        self.classes = classes
        self.probs = []
        self.labels = []

    def add(self, probs, labels):
        self.probs.append(probs)
        self.labels.append(labels)

    def value(self, return_mean=False):
        APs = {}
        labs = np.concatenate(self.labels, axis=0)
        probs = np.concatenate(self.probs, axis=0)
        for klass in range(len(self.classes)):
            positives = probs[labs == klass]
            negatives = probs[(labs != klass)]
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

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_sim
        else:
            return self._dot_sim

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_sim(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_sim(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        if zis.shape[0] != self.batch_size:
            self.batch_size = zis.shape[0]
            self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)




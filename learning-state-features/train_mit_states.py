import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from pathlib import Path
from re import L
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ActionClassifier, ResnetClassifer
from utils import AccuracyMeter
from experiment_config import ExperimentConfig, build_common_train_parser
from data.mit_states import get_mitstates_dataloaders
from data.et_action import EpicTracksAction
from data.cooking_dataset import get_cooking_dataset_dataloaders


class Experiment(ExperimentConfig):
    def __init__(self, args):
        if args.cooking_dataset:
            data_suffix = 'cooking_dataset'
        else:
            data_suffix = 'mit_states'
        super(Experiment, self).__init__(args, initialize_model=False, data_suffix=data_suffix)
        
        self.adjectives = args.adj
        if args.cooking_dataset:
            self.data = get_cooking_dataset_dataloaders(self.config["data"])
        else:
            self.data = get_mitstates_dataloaders(self.config["data"], filtered_nouns=True, adj=self.adjectives)
        
        self.model = ResnetClassifer(n_classes=self.data["train"].dataset.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["training"]["lr"])
        self.criterion = nn.CrossEntropyLoss() if not self.adjectives else nn.BCEWithLogitsLoss(reduction='none')

        if not self.adjectives:
            self.acc_meter = {split: AccuracyMeter() for split in ["train", "validation"]}

    def loop(self, split="train", num_batches=None):
        if split == "train":
            self.model.train()
        else:
            self.model.eval()
        
        for b, (images0, labels, weights) in enumerate(tqdm(self.data[split], leave=False, dynamic_ncols=True)):
            with torch.set_grad_enabled(split == 'train'):
                images0 = images0.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images0)
                # pdb.set_trace()
                loss = self.criterion(logits, labels)
                if self.adjectives:
                    # pdb.set_trace()
                    loss = (loss * weights.to(self.device)).sum(-1).mean(-1)
                # backpropagate if training
                if split == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                if not self.adjectives:
                    self.acc_meter[split].update(logits, labels)

            self.loss_meter[split].update(loss.item())
            

if __name__ == "__main__":
    parser = build_common_train_parser("./configs/mit_states.yaml")
    parser.add_argument('--adj', action='store_true', help="Train for adjectives on MIT states instead of nouns")
    parser.add_argument('--cooking_dataset', action='store_true', help="Train for adjectives on Cooking Datasets instead of mit_states")
    args = parser.parse_args()

    experiment = Experiment(args)
    experiment.run()

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ABMIL, ActionClassifier, ResnetClassifer
from data.et_action import ETABMIL, ABMILCollator, EpicTracksAction, EpicTracksCleanAction
from experiment_config import ExperimentConfig, build_common_train_parser


class Experiment(ExperimentConfig):
    def __init__(self, args):
        super(Experiment, self).__init__(args, initialize_model=False)

        self.clean = args.clean
        self.noun = args.noun
        self.abmil = args.abmil

        if self.clean:
            self.logger.info(f"Training on clean grid tracks! Nouns={self.noun}")
            self.num_samples = 5
            self.data = {
                spl: DataLoader(
                    EpicTracksCleanAction(spl, self.noun, self.num_samples),
                    batch_size=self.batch_size, shuffle=spl =="train",
                    num_workers=self.config["data"]["nw"])
                for spl in ["train", "validation"]
            }
            self.model = ResnetClassifer(len(self.data["train"].dataset.classes), pretrained=False).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
        elif self.abmil:
            self.logger.info(f"Training ABMIL!")
            self.data = {
            "train": DataLoader(
                    ETABMIL("train", self.config["data"].get("filter", None), transform_list=["color"]),#["color", "crop"]),
                            batch_size=self.batch_size, shuffle=True, num_workers=self.config["data"]["nw"], collate_fn=ABMILCollator()
                )
            }
            self.data["validation"] = DataLoader(
                ETABMIL("validation", self.config["data"].get("filter", None), transform_list=[], verb2class=self.data["train"].dataset.verb2class),
                batch_size=self.batch_size, shuffle=False, num_workers=self.config["data"]["nw"], collate_fn=ABMILCollator()
            )
            self.model = ABMIL(self.config["model"], self.device).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss() #weight=self.data["train"].dataset.weight.to(self.device))
        else:
            # set up dataloaders
            self.data = {
                spl: DataLoader(
                    EpicTracksAction(f"{spl}_{self.data_suffix}", True, self.config["data"].get("filter", None), frac=0.5),
                    batch_size=self.batch_size, shuffle=spl =="train",
                    num_workers=self.config["data"]["nw"])
                for spl in ["train", "validation"]
            }
            self.model = ActionClassifier(self.config, self.data["train"].dataset.classes).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["training"]["lr"])#, weight_decay=self.config["training"]["decay"])

    def get_loss(self, dic, split):
        if self.clean:
            outs = [self.model(dic[f"image_{i}"]) for i in range(self.num_samples)]
            losses = torch.stack([self.criterion(outs[i], dic["label"]) for i in range(self.num_samples)])
            loss = torch.min(losses)
        elif self.abmil:
            ascore, bscore = self.model(dic["btracks"], dic["atracks"], dic["bnum"], dic["anum"])
            logits = ascore + bscore
            pred_labels = logits.argmax(dim=1)
            return self.criterion(logits, dic["labels"])
        else:
            out = self.model(dic)
            probs = out['probs']
            action = dic['action']
            loss = self.criterion(probs, action)
        return loss      

    def loop(self, split="train", num_batches=None):
        is_train = "train" in split
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        t = tqdm(self.data[split], leave=False, dynamic_ncols=True)
        for b, dic in enumerate(t):
            dic = {key: value.to(self.device) for key, value in dic.items()}
            with torch.set_grad_enabled(is_train):
                loss = self.get_loss(dic, split)
                # backpropagate if training
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                t.set_postfix({"loss":loss.item()})
            self.loss_meter[split].update(loss.item())
            if b == num_batches:
                break


if __name__ == "__main__":
    parser = build_common_train_parser("./configs/action_classifier.yaml")
    parser.add_argument('--clean', action='store_true', help="Use clean track annotations")
    parser.add_argument('--noun', action='store_true', help="Predict nouns on clean tracks")
    parser.add_argument('--abmil', action='store_true', help="Train ABMIL action classification model")
    args = parser.parse_args()

    experiment = Experiment(args)
    experiment.run()

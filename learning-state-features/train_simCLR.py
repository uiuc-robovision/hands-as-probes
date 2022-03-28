import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from pathlib import Path
import argparse
from datetime import date
import pprint
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.et_simclr import EpicTracks4VanillaSimCLR
from data.epic_frame import EpicFrames
from losses import NTXentLoss
from experiment_config import ExperimentConfig, build_common_train_parser

from data.et_simclr_tcn import SimCLRwTCNCollator, EpicTracks4SimCLRwTCN


class Experiment(ExperimentConfig):
    def __init__(self, args):
        ds = "ioumf" if args.object_crops else "rgb_frames"
        super(Experiment, self).__init__(args, data_suffix=ds)

        with_tcn = self.config["data"].get("tcn", False)

        if with_tcn:
            self.logger.info("Training Temporal SimCLR with TCN")
            datasets = {
                "train": EpicTracks4SimCLRwTCN(f"train_{self.data_suffix}", self.config["data"].get("filter", None), self.segment_range),
                "validation": EpicTracks4SimCLRwTCN(f"validation_{self.data_suffix}", self.config["data"].get("filter", None), self.segment_range)
            }

            self.logger.debug(f"Before Removing (# tracks): \t Train: {len(datasets['train'])} | Val: {len(datasets['validation'])}")
            datasets["train"].remove_small_tracks(self.min_segment_len)
            datasets["validation"].remove_small_tracks(self.min_segment_len)
            self.logger.debug(f"After Removing |tracks| < {self.min_segment_len}: \t Train: {len(datasets['train'])} | Val: {len(datasets['validation'])}")

        elif args.object_crops:
            datasets = {
                "train": EpicTracks4VanillaSimCLR(f"train_{self.data_suffix}", iou_threshold=1),
                "validation": EpicTracks4VanillaSimCLR(f"validation_{self.data_suffix}", iou_threshold=1)
            }
        
        else:
            crop = self.config["data"].get("crop", None)
            assert crop is not None
            datasets = {
                "train": EpicFrames("train", crop=crop, max_track_len=128, temporal_simclr=True),
                "validation": EpicFrames("validation", crop=crop, max_track_len=128, temporal_simclr=True)
            }

        self.data = {
            key: DataLoader(val, batch_size=self.batch_size,
                            shuffle=(key == "train"),
                            num_workers=self.config["data"]["nw"],
                            pin_memory=True,
                            collate_fn=SimCLRwTCNCollator() if with_tcn else None) 
            for key, val in datasets.items()
        }

        self.criterion = NTXentLoss(self.device, self.batch_size, 0.1, True)

    def loop(self, split="train", num_batches=None):
        batch_count = 0
        if split == "train":
            self.model.train()
        else:
            self.model.eval()
        for b, (images1, images2) in enumerate(tqdm(self.data[split], leave=False, dynamic_ncols=True)):
            batch_count += 1
            images1 = images1.to(self.device)
            images2 = images2.to(self.device)
            with torch.set_grad_enabled(split == 'train'):
                emb1 = self.model(images1)
                emb2 = self.model(images2)
                loss = self.criterion(emb1, emb2)
                # backpropagate if training
                if split == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            self.loss_meter[split].update(loss.item())
            if b == num_batches:
                break

    def run(self):
        for e in range(self.start_epoch, self.config["training"]["num_epochs"]):
            # run training loop
            self.loop("train", self.config["training"]["num_batches"])
            # log interval
            if (e + 1) % self.config["training"]["log_interval"] == 0:
                train_loss = self.loss_meter["train"].average()
                self.logger.info(f'Epoch: {e}; Train Loss: {train_loss}')
                self.loss_meter["train"].reset()
            # checkpoint interval
            if (e + 1) % self.config["training"]["ckpt_interval"] == 0:
                self.loop("validation")
                val_loss = self.loss_meter["validation"].average()
                self.save(e, self.loss_meter["validation"].check() == -1)
                self.logger.info(f'Epoch: {e}; Validation Loss: {val_loss}')
                self.loss_meter["validation"].reset()
        # save checkpoint at the end of training
        self.save(e, False)


if __name__ == "__main__":
    parser = build_common_train_parser("./configs/simclr.yaml")
    parser.add_argument('--object_crops', action='store_true', help="Run on object crops instead")
    args = parser.parse_args()

    experiment = Experiment(args)
    experiment.run()

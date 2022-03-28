import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from losses import NTXentLoss
from data.et_simclr import EpicTracks4SimCLR
from experiment_config import ExperimentConfig, build_common_train_parser


class Experiment(ExperimentConfig):
    def __init__(self, args):
        super(Experiment, self).__init__(args)
        
        # set up dataloaders
        self.logger.info("Training Temporal SimCLR")
        datasets = {
            "train": EpicTracks4SimCLR(f"train_{self.data_suffix}", self.config["data"].get("filter", None), self.segment_range),
            "validation": EpicTracks4SimCLR(f"validation_{self.data_suffix}", self.config["data"].get("filter", None), self.segment_range)
        }
        datasets["train"].remove_small_tracks(self.min_segment_len)
        datasets["validation"].remove_small_tracks(self.min_segment_len)
        self.logger.debug(f"After Removing |tracks| < {self.min_segment_len}: \t Train: {len(datasets['train'])} | Val: {len(datasets['validation'])}")

        self.data = {
            key: DataLoader(val, batch_size=self.batch_size, shuffle=(key == "train"),
                            num_workers=self.config["data"]["nw"], pin_memory=True) 
            for key, val in datasets.items()
        }

        self.criterion = NTXentLoss(self.device, self.batch_size, 0.1, True)


    def loop(self, split="train", num_batches=None):
        batch_count = 0
        if split == "train":
            self.model.train()
        else:
            self.model.eval()
        t = tqdm(self.data[split], leave=False, dynamic_ncols=True)
        for b, dicts in enumerate(t):
            batch_count += 1
            images1 = dicts["pos1"].to(self.device)
            images2 = dicts["pos2"].to(self.device)
            with torch.set_grad_enabled(split == 'train'):
                emb1 = self.model(images1)
                emb2 = self.model(images2)
                loss = self.criterion(emb1, emb2)
                t.set_postfix({"loss": loss.item()})
                # backpropagate if training
                if split == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            self.loss_meter[split].update(loss.item())
            if b == num_batches:
                break

if __name__ == "__main__":
    parser = build_common_train_parser("./configs/tsc.yaml")
    args = parser.parse_args()
    
    experiment = Experiment(args)
    experiment.run()

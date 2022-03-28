import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from experiment_config import ExperimentConfig, build_common_train_parser
from utils import set_torch_seed
from model import SimCLRLayer
from data.et_tcn import EpicTracks4TCN
from losses import TripletMarginLoss


class Experiment(ExperimentConfig):
    def __init__(self, args):
        super(Experiment, self).__init__(args, initialize_model=False)

        # set up dataloaders
        datasets = {
            "train": EpicTracks4TCN(f"train_{self.data_suffix}", self.config["data"].get("filter", None),
                                    transform_list=self.config["data"].get("transforms", []),
                                    segment_len_range=(0,self.config["data"]["maxlen"])),
            "validation": EpicTracks4TCN(f"validation_{self.data_suffix}",
                                         self.config["data"].get("filter", None),
                                         segment_len_range=(0,self.config["data"]["maxlen"]))
        }

        if self.config["data"].get("sampler", "random") == "weighted":
            weights = {key: val.track_lengths for key, val in datasets.items()}
            '''
            weights["train"][weights["train"] < 12] = 0
            weights["validation"][weights["validation"] < 12] = 0
            '''
            samplers = {key: WeightedRandomSampler(weights=[weights[key][Path(tf).name] for tf in val.track_files],
                                                   num_samples=len(val.track_files), replacement=True)
                        for key, val in datasets.items()}
        else:
            samplers = {key: WeightedRandomSampler(weights=torch.ones(len(val.track_files)),
                                                   num_samples=len(val.track_files), replacement=True)
                        for key, val in datasets.items()}

        self.data = {
            key: DataLoader(val, batch_size=self.batch_size, sampler=samplers[key],
                            num_workers=self.config["data"]["nw"]) for key, val in datasets.items()}

        self.criterion = TripletMarginLoss()
        self.model = SimCLRLayer(512, 128, normalize=False, alpha=0.0).to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config["training"]["lr"])

    def loop(self, split="train", num_batches=None):
        batch_count = 0
        if split == "train":
            self.model.train()
        else:
            self.model.eval()
        for b, (aimg, pimg, nimg) in enumerate(tqdm(self.data[split], leave=False, dynamic_ncols=True)):
            batch_count += 1
            imgs = torch.cat([aimg, pimg, nimg], dim=0)
            imgs = imgs.to(self.device)
            with torch.set_grad_enabled(split == 'train'):
                emb = self.model(imgs).reshape(3, aimg.shape[0], -1)
                loss = self.criterion(emb[0], emb[1], emb[2])
                # backpropagate if training
                if split == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            self.loss_meter[split].update(loss.item())
            if b == num_batches:
                break

if __name__ == "__main__":
    parser = build_common_train_parser("./configs/tcn.yaml")
    args = parser.parse_args()

    experiment = Experiment(args)
    experiment.run()

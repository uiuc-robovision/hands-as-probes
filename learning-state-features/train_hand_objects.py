import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from tqdm import tqdm

import torch
import torch.nn
from torch.utils.data import DataLoader
from experiment_config import ExperimentConfig, build_common_train_parser
import torch.optim as optim

from losses import NTXentLoss
from model import SimCLRLayerMultiHead, SimCLRLayerWithPositionalEncoding
from data.et_hand_objects import EpicHandObjectsTracks


class Experiment(ExperimentConfig):
    def __init__(self, args):
        super(Experiment, self).__init__(args, initialize_model=False)
        
        self.hand_appearance = self.config["training"]["add_hand_appearance"]
        self.hand_motion = self.config["training"]["add_hand_motion"]

        datasets = {
            split: EpicHandObjectsTracks(
                f"{split}_ioumf", 
                self.config,
                self.segment_range
            )
            for split in ["train", "validation"]
        }

        self.data = {
            key: DataLoader(val, batch_size=self.batch_size, shuffle=(key == "train"),
                            num_workers=self.config["data"]["nw"], pin_memory=True) 
                            for key, val in datasets.items()
        }

        if self.hand_appearance and not self.hand_motion:
            self.logger.info("Adding corresponding hand image only!")
            self.model = SimCLRLayerMultiHead(512, 128, num_heads=2)
            self.model_oh = SimCLRLayerMultiHead(512, 128, num_heads=1)
        elif self.hand_appearance and self.hand_motion:
            self.logger.info("Adding hand appearance with motion!")
            self.model = SimCLRLayerMultiHead(512, 128, num_heads=2)
            self.model_oh = SimCLRLayerWithPositionalEncoding(512, 128, num_objects=1)
        elif not self.hand_appearance and self.hand_motion:
            self.logger.info("Adding only hand motion (no appearance).")
            self.model = SimCLRLayerMultiHead(512, 128, num_heads=2)
            self.model_oh = SimCLRLayerWithPositionalEncoding(512, 128, num_objects=3, only_pe=True)
        else:
            assert False, "Configuration not supported."

        self.model = self.model.to(self.device)
        self.model_oh = self.model_oh.to(self.device)
        model_params = list(filter(lambda p: p .requires_grad, self.model.parameters()))
        model_params += list(filter(lambda p: p .requires_grad, self.model_oh.parameters()))
        self.optimizer = optim.Adam(model_params, self.config["training"]["lr"])
        self.criterion = NTXentLoss(self.device, self.batch_size, 0.1, True)

        # try to restore checkpoint
        if self.config["training"].get("resume", True):
            self.resume()
            self.logger.info(f"Loaded model. Resuming training from epoch {self.start_epoch}")
        else:
            self.logger.info(f"Training model from scratch")


    def save(self, epoch, best):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'model_object_hands_state_dict': self.model_oh.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.loss_meter["validation"].lowest,
            'validation_loss': self.loss_meter["validation"].avg_list,
            'train_loss': self.loss_meter["train"].avg_list,
            'config': self.config
        }
        if (epoch + 1) % 50 == 0:
            torch.save(save_dict, self.model_dir / f"checkpoint_{epoch + 1}.pth")
        torch.save(save_dict, self.model_dir / f"checkpoint_last.pth")
        if best:
            torch.save(save_dict, self.model_dir / f"checkpoint_best.pth")

    def resume(self):
        model_name = self.config["training"].get("resume_name", f"checkpoint_last.pth")
        checkpoint = torch.load(self.model_dir / model_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model_oh.load_state_dict(checkpoint["model_object_hands_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_meter["validation"].lowest = checkpoint["best_loss"]
        self.loss_meter["train"].avg_list = checkpoint["train_loss"]
        self.loss_meter["validation"].avg_list = checkpoint["validation_loss"]
        self.config = checkpoint["config"]
        self.start_epoch = checkpoint["epoch"] + 1

    def loop(self, split="train", num_batches=None):
        if split == "train":
            self.model.set(train=True)
            self.model_oh.set(train=True)
        else:
            self.model.set(train=False)
            self.model_oh.set(train=False)

        t = tqdm(self.data[split], leave=False, dynamic_ncols=True)
        for b, dicts in enumerate(t):
            loss_tsc, loss_hand_match = torch.tensor(0), torch.tensor(0)
            w_tsc, w_hand_match = 0.5, 0.5

            with torch.set_grad_enabled(split == 'train'):
                images_op1 = dicts["object_pos1"].to(self.device)
                images_op2 = dicts["object_pos2"].to(self.device)
                emb_op1 = self.model(images_op1, head_idx=0)
                emb_op2 = self.model(images_op2, head_idx=0)
                loss_tsc = self.criterion(emb_op1, emb_op2)

                images_oh_o = dicts["object_hand_obj"].to(self.device)
                emb_ohp1 = self.model(images_oh_o, head_idx=1)

                images_oh_h1 = dicts["object_hand_hand"].to(self.device)
                if self.hand_motion:
                    emb_ohp2 = self.model_oh(images_oh_h1, dicts["hand_pose"].to(self.device), head_idx=0)
                else:
                    emb_ohp2 = self.model_oh(images_oh_h1, head_idx=0)
                
                loss_hand_match = self.criterion(emb_ohp1, emb_ohp2)

                loss = w_tsc*loss_tsc + w_hand_match*loss_hand_match
                
                debug_dict = {
                    "loss_tsc": loss_tsc.item(), 
                    "loss_hand": loss_hand_match.item(),
                    "loss": loss.item(),
                }
                t.set_postfix(debug_dict)

                if split == "train":
                    self.optimizer.zero_grad(True)
                    loss.backward()
                    self.optimizer.step()
            
            self.loss_meter[split].update(loss.item())
            if b == num_batches:
                break


if __name__ == "__main__":
    parser = build_common_train_parser("./configs/hand_objects_full.yaml")
    args = parser.parse_args()

    experiment = Experiment(args)
    experiment.run()

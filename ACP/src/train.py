import os, sys
from helper import set_numpythreads
set_numpythreads()

import numpy as np
import argparse
import pprint
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils import set_torch_seed, load_config, Meter, Logger, APMeterBin
from model import SegmentationNetDeeper, SegmentationNetDeeperBig
# from data.Base import EPICPatchLoader, worker_init_fn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, 1048576))

class Experiment:
    def __init__(self, config_path, model_dir, logdir, name):
        self.name = name
        self.model_dir = model_dir
        self.logdir = logdir
        self.masterlogger = Logger("main", os.path.join(self.logdir, f"{self.name}.log"))
        self.logger = self.masterlogger.logger

        self.config = load_config(config_path)
        
        if self.config["training"].get("resume", True):
            try:
                checkpoint = torch.load(os.path.join(model_dir, f"{self.name}_checkpoint.pth"))
                self.config = checkpoint["config"]
                self.config['training']['resume'] = True
            except OSError as e:
                self.logger.error(e)
                self.logger.info("Starting training from scratch")
        else:
            self.logger.info("resume training flag set to false")

        self.logger.info(pprint.pformat(self.config))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = self.config["data"]["bs"]
        self.rng = np.random.default_rng(0)

        self.is_sym = self.config['model'].get("is_sym", False)
        if self.is_sym:
            self.logger.info("Using symmetric encoder-decoder architecture")
            self.loss_masking = self.config['training']['loss_masking']
            self.logger.info(f"Loss Masking {self.loss_masking}")

        # Set dataset and dataloaders
        self.set_data_loader()

        if not self.is_sym:
            self.model = SegmentationNetDeeper(self.config['model']).to(self.device)
        else:
            self.model = SegmentationNetDeeperBig(self.config['model']).to(self.device)
        pos_weight = self.config['training'].get("pos_weight", 4.)
        self.logger.info(f"Pos Weight set to {pos_weight}")
        self.weights = torch.from_numpy(np.array([1, pos_weight])).type(torch.float32).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.weights[1])

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["training"]["lr"], weight_decay=self.config["training"]["decay"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=3, threshold=0.005, min_lr=1e-5, verbose=True)
        self.loss_meter = {"train": Meter(), "validation": Meter()}
        self.acc_meter = {"train": Meter(), "validation": Meter()}
        self.mAP_meter = {"train": APMeterBin(), "validation": APMeterBin()}
        self.start_epoch = 0

        # try to restore checkpoint
        if self.config["training"].get("resume", True):
            try:
                checkpoint = torch.load(os.path.join(model_dir, f"{self.name}_checkpoint.pth"))
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                self.loss_meter["validation"].lowest = checkpoint["best_loss"]
                self.loss_meter["train"].avg_list = checkpoint["train_loss"]
                self.loss_meter["validation"].avg_list = checkpoint["validation_loss"]
                self.config = checkpoint["config"]
                self.start_epoch = checkpoint["epoch"] + 1
                self.logger.info("loaded model ...")
            except OSError as e:
                self.logger.error(e)
                self.logger.info("Starting training from scratch")
        else:
            self.logger.info("resume training flag set to false")


        self.train_writer = SummaryWriter(os.path.join(self.logdir, self.name))
    
    def set_data_loader(self):
        global worker_init_fn
        if self.config['training']['use_hand_segmask']:
            self.logger.info("Using Hand Segmentation Masks for training")
            assert not self.is_sym
            from data.Base_seg import EPICPatchLoaderSeg, worker_init_fn
            self.datasets = {
                "train": EPICPatchLoaderSeg(self.config,
                                        split="train",
                                        transform=True,
                                    ),
                "validation": EPICPatchLoaderSeg(self.config,
                                        split="validation",
                                        transform=False, length=10000,
                                    )
            }
        elif self.is_sym:
            self.logger.info("Using symmetric dataloader for training")
            from data.Base_inpoutmatch import EPICPatchLoaderInpOutMatch, worker_init_fn
            self.datasets = {
                "train": EPICPatchLoaderInpOutMatch(self.config,
                                        split="train",
                                        transform=True,
                                    ),
                "validation": EPICPatchLoaderInpOutMatch(self.config,
                                        split="validation",
                                        transform=False, length=10000,
                                    )
            }
        else:
            from data.Base import EPICPatchLoader, worker_init_fn
            self.datasets = {
                "train": EPICPatchLoader(self.config,
                                        split="train",
                                        transform=True,
                                    ),
                "validation": EPICPatchLoader(self.config,
                                        split="validation",
                                        transform=False, length=10000,
                                    )
            }
        
        # set up dataloaders
        self.data = {
            "train": DataLoader(self.datasets["train"],
                batch_size=self.batch_size, shuffle=True,
                num_workers=self.config["data"]["nw"],
                worker_init_fn=worker_init_fn,
                drop_last=False),
            "validation": DataLoader(self.datasets["validation"],
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.config["data"]["nw"],
                worker_init_fn=worker_init_fn,
                drop_last=False),
        }


        return

    def create_train_dataloader(self, num_samples):
        probs = np.ones((len(self.datasets['train'])))/len(self.datasets['train'])
        if self.config['training'].get('length_based_sampling', False):
            uprobs = np.array(self.datasets['train'].probs)
            probs = uprobs / np.sum(uprobs)
        if self.config['training'].get('replace', False):
            indices = self.rng.choice(len(self.datasets['train']),
                                    size=num_samples * self.config['data']['bs'],
                                    replace=True,
                                    p=probs)
        else:
            indices = self.rng.choice(len(self.datasets['train']),
                                    size=min(num_samples * self.config['data']['bs'], len(self.datasets['train'])),
                                    replace=False,
                                    p=probs)
        self.data["train"] = DataLoader(
                self.datasets['train'],
                batch_size=self.config['data']['bs'], shuffle=False,
                num_workers=self.config['data']['nw'],
                sampler=SubsetRandomSampler(indices),
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                drop_last=True)


    def save(self, epoch, best):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.loss_meter["validation"].lowest,
            'validation_loss': self.loss_meter["validation"].avg_list,
            'train_loss': self.loss_meter["train"].avg_list,
            'config': self.config
        }
        torch.save(save_dict, os.path.join(self.model_dir, f"{self.name}_checkpoint.pth"))

        if self.config["training"].get("save_after", None) is not None:
            if (epoch + 1) % (self.config["training"].get("save_after", None) * self.config["training"]["ckpt_interval"]) == 0:
                torch.save(save_dict, os.path.join(self.model_dir, f"{self.name}_checkpoint_{epoch + 1}.pth"))                

        if best:
            torch.save(save_dict, os.path.join(self.model_dir, f"{self.name}_best.pth"))

    def get_loss(self, dic, split):

        out = self.model(dic['img'])

        preds = out['pred']
        probs = torch.sigmoid(preds)

        seg_mask_exp = dic['seg_mask']
        
        if self.is_sym:
            vmask = dic['valid_mask']
            if self.loss_masking:
                # Use inverted validation mask to mask the loss
                loss_seg2d = self.criterion(preds, seg_mask_exp)*(1 - vmask)
            else:
                # Do not mask the loss
                loss_seg2d = self.criterion(preds, seg_mask_exp)
        else:
            loss_seg2d = self.criterion(preds, seg_mask_exp)

        loss_seg = loss_seg2d.mean(-1).mean(-1)
        loss_seg = torch.mean(loss_seg)

        predictions = preds.reshape(-1, 1)
        labels = dic['seg_mask'].reshape(-1, 1)

        acc = torch.mean(((torch.sigmoid(predictions) > 0.5)*1.0 == labels)*1.0)
        self.acc_meter[split].update(acc)
        self.mAP_meter[split].add(predictions.detach().cpu().numpy(), labels.detach().cpu().numpy())

        return loss_seg, probs, dic['seg_mask'], dic['seg_image']

    def loop(self, split="train", num_batches=None, epoch=0):
        if 'train' in split:
            self.model.train()
            if num_batches is not None:
                self.create_train_dataloader(num_batches)
        else:
            self.model.eval()

        for i, dic in tqdm(enumerate(self.data[split]), total=len(self.data[split])):
            dic = {key: value.cuda() for key, value in dic.items()}
            with torch.set_grad_enabled('train' in split):
                loss, preds, masks, img = self.get_loss(dic, split)
                # backpropagate if training
                if 'train' in split:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if i == 0:
                    self.train_writer.add_images(f'{split}_gtimg', img, epoch)
                    self.train_writer.add_images(f'{split}_preds', preds, epoch)
                    self.train_writer.add_images(f'{split}_masks', masks, epoch)
            self.loss_meter[split].update(loss.item())

    def run(self):
        for e in range(self.start_epoch, self.start_epoch + self.config["training"]["num_epochs"]):

            # run training loop
            self.loop("train", self.config["training"]["num_batches"], epoch=e)
            
            # log interval
            if (e + 1) % self.config["training"]["log_interval"] == 0:
                train_loss = self.loss_meter["train"].average()
                train_acc = self.acc_meter["train"].average()
                train_seg_mAP = self.mAP_meter["train"].value_random(20)
                
                self.scheduler.step(train_loss)
                self.train_writer.add_scalar('training_loss', train_loss, e)
                self.train_writer.add_scalar('training_acc', train_acc, e)

                self.logger.info(f'Epoch: {e+1}; Train Loss: {train_loss:.3f}; Train Acc: {train_acc:.3f} Train seg AP {train_seg_mAP:.3f}')
            self.loss_meter["train"].reset()
            self.acc_meter["train"].reset()
            self.mAP_meter["train"].reset()
            
            # checkpoint interval
            if (e + 1) % self.config["training"]["ckpt_interval"] == 0:
                self.loop("validation", epoch=e)
                val_loss = self.loss_meter["validation"].average()
                val_acc = self.acc_meter["validation"].average()
                val_seg_mAP = self.mAP_meter["validation"].value_random(20)
                
                self.train_writer.add_scalar('validation_loss', val_loss, e)
                self.train_writer.add_scalar('validation_acc', val_acc, e)
                
                self.save(e, self.loss_meter["validation"].check(highest=False) == -1)
                
                self.logger.info(f'Epoch: {e+1}; Validation Loss: {val_loss:.3f}; Validation Acc: {val_acc:.3f} Validation Seg AP: {val_seg_mAP:.3f}')
                
                self.loss_meter["validation"].reset()
                self.acc_meter["validation"].reset()
                self.mAP_meter["validation"].reset()

        # save checkpoint at the end of training
        self.save(e, False)

        self.masterlogger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training hyper-parameters')
    
    parser.add_argument('--config', dest='config', type=str,
                        default="./src/configs/base.yaml")
    parser.add_argument('--model', dest='model', type=str,
                        default="./models/EPIC-KITCHENS/")
    parser.add_argument('--log', dest='log', type=str,
                        default="./logs/EPIC-KITCHENS/")
    parser.add_argument('--name', dest='name', type=str, default="SegNetDeeper18_multi5")
    parser.add_argument('--seed', dest='seed', type=int,
                        default=0)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False)
    

    args = parser.parse_args()
    os.makedirs(args.model, exist_ok=True)
    os.makedirs(args.log, exist_ok=True)
    
    set_torch_seed(args.seed)
    if args.shuffle:
        args.name = args.name + f"_shuffle"
    args.name = args.name + f"_seed{args.seed}"

    experiment = Experiment(args.config, args.model, args.log, args.name)
    if args.shuffle:
        experiment.rng = np.random.default_rng(args.seed)
    experiment.run()

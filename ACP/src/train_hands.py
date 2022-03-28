import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from helper import set_numpythreads
set_numpythreads()

import numpy as np
import pandas as pd
import argparse
import pprint
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils import set_torch_seed, load_config, Meter, Logger, APMeterBin, create_imagewtext
from model import SegmentationNetDeeperTwohead, SegmentationNetDeeperTwoheadBig
from model_grasp import ClassificationNet, SimCLRwGUN71
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

        metadata = pd.read_pickle(args.GUN71_annotations)
        hand_classes = metadata['grasp'].value_counts().sort_index().index
        class2id = {j: i for i, j in enumerate(hand_classes)}

        self.use_class_loss = self.config['training']["use_class_loss"]
        if self.use_class_loss:
            grasp_info = load_config(args.grasp_info)
            sel_classes = grasp_info['easy_classes']['classes']
            sel_names = grasp_info['easy_classes']['names']
            
            self.filt_ids = []
            filt_names = []

            for c, n in zip(sel_classes, sel_names):
                if c != 'None':
                    self.filt_ids.append(class2id[c])
                    filt_names.append(n)
            num_clusters = len(self.filt_ids)
            self.cluster_images = [create_imagewtext((128, 128), n, 30, RGB=True) for n, id in zip(filt_names, self.filt_ids)]
            self.cluster_images = [torch.from_numpy(np.array(i).astype(np.float32)/255).permute(2, 0, 1) for i in self.cluster_images]

        self.rescale_counts = torch.ones(num_clusters).to(self.device)
        # Load the hand grasp model
        best_model = torch.load(self.config['training']['hand_ckpt'])
        if "gun71_tsc" in os.path.basename(self.config['training']['hand_ckpt']).lower():
            self.hand_model = SimCLRwGUN71(nclasses=len(hand_classes), model_config=best_model["config"]['model']).to(self.device)
        else:

            self.hand_model = ClassificationNet(n_classes=len(hand_classes)).to(self.device)
        mweights = best_model["model_state_dict"]
        self.hand_model.load_state_dict(mweights)
        
        # Freeze weights and set the model in eval mode
        self.hand_model.eval()
        for param in self.hand_model.parameters():
            param.requires_grad = False

        if not self.is_sym:
            self.model = SegmentationNetDeeperTwohead(self.config['model'], emb_size=num_clusters).to(self.device)
        else:
            self.model = SegmentationNetDeeperTwoheadBig(self.config['model'], emb_size=num_clusters).to(self.device)
        pos_weight = self.config['training'].get("pos_weight", 4.)
        self.logger.info(f"Pos Weight set to {pos_weight}")
        self.weights = torch.from_numpy(np.array([1, pos_weight])).type(torch.float32).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=self.weights[1])
        self.accriterion = torch.nn.CrossEntropyLoss(reduction="none")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["training"]["lr"], weight_decay=self.config["training"]["decay"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=3, threshold=0.005, min_lr=1e-5, verbose=True)
        self.loss_meter = {"train": Meter(), "validation": Meter()}
        self.acc_meter = {"train": Meter(), "validation": Meter()}
        self.ac_loss_meter = {"train": Meter(), "validation": Meter()}
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
            from data.Hands_seg import EPICPatchLoaderSegwHands, worker_init_fn
            self.datasets = {
                "train": EPICPatchLoaderSegwHands(self.config,
                                        split="train",
                                        transform=True,
                                    ),
                "validation": EPICPatchLoaderSegwHands(self.config,
                                        split="validation",
                                        transform=False, length=10000,
                                    )
            }
        elif self.is_sym:
            self.logger.info("Using symmetric dataloader for training")
            from data.Hands_inpoutmatch import EPICPatchLoaderInpOutMatchwHands, worker_init_fn
            self.datasets = {
                "train": EPICPatchLoaderInpOutMatchwHands(self.config,
                                        split="train",
                                        transform=True,
                                    ),
                "validation": EPICPatchLoaderInpOutMatchwHands(self.config,
                                        split="validation",
                                        transform=False, length=10000,
                                    )
            }
        else:
            from data.Hands import EPICPatchLoaderwHands, worker_init_fn
            self.datasets = {
                "train": EPICPatchLoaderwHands(self.config,
                                        split="train",
                                        transform=True,
                                    ),
                "validation": EPICPatchLoaderwHands(self.config,
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
                pin_memory=True,
                drop_last=False),
            "validation": DataLoader(self.datasets["validation"],
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.config["data"]["nw"],
                worker_init_fn=worker_init_fn,
                pin_memory=True,
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

        hand_indices = dic['hand_sampled'] == 1
        hemb_pred = out['hand_emb'][hand_indices]

        # Init Hand Loss
        hand_loss = 0

        if len(hemb_pred) > 0:
            with torch.no_grad():
                probs_unscaled = self.hand_model.forward_classifier(dic['hand'][hand_indices])[:, self.filt_ids]
                # scores = torch.sigmoid(probs_unscaled)
                scores = F.softmax(probs_unscaled, dim=1)

                # max_sc, hand_labels = scores.max(dim=1)
                max_sc_k, hand_posits = scores.topk(k=self.config["training"]["topkclass"], dim=1, largest=True)
                min_sc_k, hand_negs = scores.topk(k=self.config["training"]["bottomkclass"], dim=1, largest=False)

                final_hand_labels = torch.zeros_like(scores)
                final_hand_labels[torch.arange(len(max_sc_k)).unsqueeze(1).repeat(1, hand_posits.shape[1]), hand_posits] = 1
                final_hand_labels[torch.arange(len(max_sc_k)).unsqueeze(1).repeat(1, hand_negs.shape[1]), hand_negs] = 0

                final_hand_mask = torch.zeros_like(scores)
                final_hand_mask[torch.arange(len(max_sc_k)).unsqueeze(1).repeat(1, hand_posits.shape[1]), hand_posits] = 1
                final_hand_mask[torch.arange(len(max_sc_k)).unsqueeze(1).repeat(1, hand_negs.shape[1]), hand_negs] = 1

            un, counts = hand_posits.unique(return_counts=True, sorted=True)
            for ind, u in enumerate(un):
                self.rescale_counts[u] += counts[ind]

            norm_counts = 1. / self.rescale_counts
            norm_counts = norm_counts / torch.sum(norm_counts) * len(norm_counts)
            
            hand_loss = F.binary_cross_entropy_with_logits(
                            hemb_pred,
                            final_hand_labels,
                            weight=norm_counts,
                            reduction='none'
                            ) * final_hand_mask
            hand_loss = hand_loss.sum(1).mean(0)

        self.ac_loss_meter[split].update((loss_seg * 0 + hand_loss).item())
        loss = loss_seg + 0.5 * hand_loss
        # self.mAP_meter[split].add(predictions.detach().cpu().numpy(), labels.detach().cpu().numpy())
        if len(hemb_pred) > 0:
            return loss, probs,\
                    dic['seg_mask'], dic['seg_image'], dic['ohand'][hand_indices], hand_posits[:, 0:1]
        
        return loss, probs,\
            dic['seg_mask'], dic['seg_image'], None, None

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
                loss, preds, masks, img, hand, label = self.get_loss(dic, split)
                # backpropagate if training
                if 'train' in split:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if i == 0:
                    self.train_writer.add_images(f'{split}_gtimg', img, epoch)
                    self.train_writer.add_images(f'{split}_preds', preds, epoch)
                    self.train_writer.add_images(f'{split}_masks', masks, epoch)
                    if hand is not None:
                        labelled_hands = torch.stack([torch.cat([i, self.cluster_images[j]], dim=2) for i, j in zip(hand.cpu(), label)])
                    else:
                        labelled_hands = torch.stack(self.cluster_images)
                    self.train_writer.add_images(f'{split}_labelledhands', labelled_hands, epoch)
            self.loss_meter[split].update(loss.item())

    def run(self):
        for e in range(self.start_epoch, self.start_epoch + self.config["training"]["num_epochs"]):

            # run training loop
            self.loop("train", self.config["training"]["num_batches"], epoch=e)
            
            # log interval
            if (e + 1) % self.config["training"]["log_interval"] == 0:
                train_loss = self.loss_meter["train"].average()
                train_acc = self.acc_meter["train"].average()
                train_hand_loss = self.ac_loss_meter["train"].average()
                train_seg_mAP = self.mAP_meter["train"].value_random(20)
                
                self.scheduler.step(train_loss)
                self.train_writer.add_scalar('training_loss', train_loss, e)
                self.train_writer.add_scalar('training_acc', train_acc, e)
                self.train_writer.add_scalar('training_hand_loss', train_hand_loss, e)

                self.logger.info(f'Epoch: {e+1}; Train Loss: {train_loss:.3f}; Train Acc: {train_acc:.3f} Train seg AP {train_seg_mAP:.3f} Train hand Loss: {train_hand_loss:.3f}')
                self.logger.info(f'Counts {self.rescale_counts}')
            self.loss_meter["train"].reset()
            self.acc_meter["train"].reset()
            self.mAP_meter["train"].reset()
            self.ac_loss_meter["train"].reset()
            
            # checkpoint interval
            if (e + 1) % self.config["training"]["ckpt_interval"] == 0:
                self.loop("validation", epoch=e)
                val_loss = self.loss_meter["validation"].average()
                val_acc = self.acc_meter["validation"].average()
                val_hand_loss = self.ac_loss_meter["validation"].average()
                val_seg_mAP = self.mAP_meter["validation"].value_random(20)
                
                self.train_writer.add_scalar('validation_loss', val_loss, e)
                self.train_writer.add_scalar('validation_acc', val_acc, e)
                self.train_writer.add_scalar('validation_hand_loss', val_hand_loss)
                
                self.save(e, self.loss_meter["validation"].check(highest=False) == -1)
                
                self.logger.info(f'Epoch: {e+1}; Val Loss: {val_loss:.3f}; Val Acc: {val_acc:.3f} Val Seg AP: {val_seg_mAP:.3f} Val Hand Loss: {val_hand_loss:.3f}')
                
                self.loss_meter["validation"].reset()
                self.acc_meter["validation"].reset()
                self.mAP_meter["validation"].reset()
                self.ac_loss_meter["validation"].reset()

        # save checkpoint at the end of training
        self.save(e, False)

        self.masterlogger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training hyper-parameters')
    
    parser.add_argument('--config', dest='config', type=str,
                        default="./src/configs/base_hands.yaml")
    parser.add_argument('--model', dest='model', type=str,
                        default="./models/EPIC-KITCHENS/")
    parser.add_argument('--log', dest='log', type=str,
                        default="./logs/EPIC-KITCHENS/")
    parser.add_argument('--grasp_info', dest='grasp_info', type=str,
						default="./src/configs/grasp_info.yaml")
    parser.add_argument('--GUN71_annotations', dest='GUN71_annotations', type=str,
						default="./src/metadata/GUN71_train.pkl")
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

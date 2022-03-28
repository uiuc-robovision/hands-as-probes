import os, pdb
import shutil
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse
from datetime import date
import pprint
from tqdm import tqdm
from pathlib import Path
import logging
import json, pickle
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils import load_config, Meter, Logger, APMeterCustom
from model import VOS, FeatureExtractorMultiHead, StateClassifier, TCN, HeadStateClassifier, FeatureExtractor, ActionClassifier, FeatureExtractorBN
from model import StateClassifierwithResNet, HeadStateClassifierwithResNet
from data import StateDataLoader

torch.set_num_threads(2)

class Experiment:
    def __init__(self, log_dir, args, seed, lr=None):
        self.base_name = args.name
        self.name = f"{args.name}_seed{seed}"
        self.config = load_config(args.config)
        self.only_resnet = str(args.model_path).split("_")[0] in ["resnet", "resnet50"]
        if args.model_path.exists() or self.only_resnet:
            self.model_path = args.model_path
        else:
            self.model_path = self.config["path"][str(args.model_path)]
        
        self.model_dir = args.model_dir
        self.data_dir = args.data_dir
        self.masterlogger = Logger("main", str(log_dir / f"{self.name}.log"))
        self.logger = self.masterlogger.logger
        self.config['training']['percent'] = args.percent
        self.config['training']['sample_opposite'] = args.sample_opposite
        self.lr = lr
        if lr is not None:
            self.logger.info(f"===NEW RUN===")
            self.logger.info(f"Changing learning rate {lr}")
            self.config["training"]["lr"] = lr
        self.logger.info("Training Config")
        self.logger.info(pprint.pformat(self.config['training']))
        self.logger.info("Data Config")
        self.logger.info(pprint.pformat(self.config['data']))

        self.device = torch.device(args.gpu)
        self.sample_opposite = self.config['training']['sample_opposite']
        self.percent = self.config['training']['percent']
        batch_size = self.config["data"]["bs"]

        dic_opp = {
            'open': ['close'],
            'close': ['open'],
            'inhand': ['outofhand'],
            'outofhand': ['inhand'],
            'peeled': ['unpeeled'],
            'unpeeled': ['peeled'],
            'whole': ['cut'],
            'cut': ['whole'],
            'cooked': ['raw'],
            'raw': ['cooked'],
        }

        classes = list(dic_opp.keys())
        classes.sort()
        self.classes = classes

        class2num = {j: i for i,j in enumerate(classes)}
        self.dic_num = {}
        for key in dic_opp.keys():
            k = class2num[key]
            v = [class2num[i] for i in dic_opp[key]]
            self.dic_num[k] = v

        # set up dataloaders
        self.datasets = {
            "train": StateDataLoader(self.logger, self.config, self.data_dir, "train", args.use_aug, dic_opp,
                                    sample_opposite=self.sample_opposite,
                                    percent=self.percent,
                                    seed=seed),
            "validation": StateDataLoader(self.logger, self.config, self.data_dir, "validation", False, dic_opp),
            "test": StateDataLoader(self.logger, self.config, self.data_dir, "test", False, dic_opp)
        }

        self.data = {
            "train": DataLoader(self.datasets["train"],
                batch_size=batch_size, shuffle=True,
                num_workers=self.config["data"]["nw"],
                drop_last=False),
            "validation": DataLoader(self.datasets["validation"],
                batch_size=batch_size, shuffle=False,
                num_workers=self.config["data"]["nw"],
                drop_last=False),
            "test": DataLoader(self.datasets["test"],
                batch_size=batch_size, shuffle=False,
                num_workers=self.config["data"]["nw"],
                drop_last=False)
        }

        self.rng = np.random.default_rng(0)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

        if self.only_resnet:
            use_resnet50 = str(self.model_path).split('_')[0] == "resnet50"
            assert not (use_resnet50 and args.ensemble_res)
            model = StateClassifierwithResNet if args.ensemble_res else StateClassifier
            self.model = model(n_classes=len(classes), pretrained=True, ftune=args.ftune, resnet50=use_resnet50)
        elif 'mit_states' in str(self.model_path):
            model = StateClassifierwithResNet if args.ensemble_res else StateClassifier
            self.model = model(n_classes=len(classes), pretrained=True, ftune=args.ftune)
            model_info = torch.load(self.model_path, map_location=self.device)
            model_weights = {k.split('trunk.')[1]:v for k,v in model_info['model_state_dict'].items() if 'trunk' in k}
            self.model.trunk.load_state_dict(model_weights)
        else:
            model_info = torch.load(self.model_path, map_location=self.device)
            basemodel = FeatureExtractor(model_info["config"]["model"]["emb_size"])
            if "model_object_hands_state_dict" in model_info:
                num_heads = -1
                for i in range(10):
                    if f'network.head.{i}.0.weight' in model_info["model_state_dict"]:
                        num_heads = i+1
                        break
                basemodel = FeatureExtractorMultiHead(model_info["config"]["model"]["emb_size"], num_heads=num_heads)
            model_weights = {k: model_info["model_state_dict"]["network." + k] for k in basemodel.state_dict().keys()}
            basemodel.load_state_dict(model_weights)
            emb_size = model_info["config"]["model"]["emb_size"] if args.use_head else 512
            trunk = basemodel if args.use_head else basemodel.trunk
            model = HeadStateClassifierwithResNet if args.ensemble_res else HeadStateClassifier
            self.model = model(trunk=trunk, feat_size=emb_size, n_classes=len(self.classes), ftune=args.ftune)

        self.model = self.model.to(self.device)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config["training"]["lr"], weight_decay=self.config["training"]["decay"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=10, threshold=1e-3, min_lr=1e-5, cooldown=5, verbose=True)
        self.loss_meter = {"train": Meter(), "validation": Meter(), "test": Meter()}
        self.ap_meter = {"train": Meter(), "validation": Meter(), "test": Meter()}
        self.start_epoch = 0


    def create_train_dataloader(self, num_samples):
        probs = np.ones((len(self.datasets['train'])))/len(self.datasets['train'])
        if self.config['training'].get('length_based_sampling', False):
            uprobs = np.array(self.datasets['train'].lengths)
            uprobs[uprobs > 50] = 50
            uprobs[uprobs <= 10] = 0
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
                pin_memory=True,
                drop_last=False)

    @staticmethod
    def bce_multilabel_independent_loss(pred, lab, flag, dic):
        if len(pred[flag==0]) != 0:
            loss = (F.binary_cross_entropy(pred, lab, reduction='none') * dic['weights']).sum(-1).mean(-1)
        else:
            num_pos = 0
            loss_pos = 0
            num_neg = 0
            loss_neg = 0

            # For positive examples:
            ind = flag == 1
            if len(pred[ind]) != 0:
                pred_pos = pred[ind]
                lab_pos = lab[ind]
                num_pos = len(lab_pos)
                loss_pos = - torch.mean(torch.sum(lab_pos * torch.log(pred_pos + 1e-8), dim=1))

            # For negative examples:
            ind = flag == -1
            if len(pred[ind]) != 0:
                pred_neg = pred[ind]
                lab_neg = lab[ind]
                num_neg = len(lab_neg)
                loss_neg = - torch.mean(torch.sum(lab_neg * torch.log(1 - pred_neg + 1e-8), dim=1))

            loss = (num_pos * loss_pos + num_neg * loss_neg) / (num_pos + num_neg)

        return loss

    @staticmethod
    def get_loss(out, dic, split, meter, percent):

        prob = out['probs']
        lab = dic['label'].float()

        loss = Experiment.bce_multilabel_independent_loss(prob, lab, dic['type'], dic)
        if (split == "train") and (percent > 0):
            inds = dic['type'] == 1
            prob_pos = prob[inds]
            lab_pos = lab[inds]
            meter.add(prob_pos.detach().cpu().numpy(), lab_pos.detach().cpu().numpy())
        else:
            meter.add(prob.detach().cpu().numpy(), lab.detach().cpu().numpy())

        return loss

    def loop(self, split="train", num_batches=None):
        if 'train' in split:
            self.model.train()
            self.model.set() # Set feature extractor in eval mode
        else:
            self.model.eval()
            self.model.set() # Set feature extractor in eval mode

        meter = APMeterCustom(self.dic_num, self.classes)
        # for b, dic in enumerate(tqdm(self.data[split], leave=False, dynamic_ncols=True, desc=split)):
        for b, dic in enumerate(self.data[split]):
            dic = {key: value.to(self.device) for key, value in dic.items()}
            with torch.set_grad_enabled('train' in split):
                out = self.model(dic)
                loss = self.get_loss(out, dic, split, meter, self.percent)
                if 'train' in split:
                    self.optimizer.zero_grad(True)
                    loss.backward()
                    self.optimizer.step()
            self.loss_meter[split].update(loss.item())
        self.ap_meter[split].update(meter.value(self.sample_opposite, return_mean=True))
        
        return meter

    def run(self):
        num_epochs = self.config['training']['num_epochs']
        for e in range(self.start_epoch, num_epochs):
            # run training loop
            train_meter = self.loop("train", self.config["training"]["num_batches"])
            train_loss = self.loss_meter["train"].average()
            trainmAP = self.ap_meter["train"].average()
            self.scheduler.step(train_loss)
            # log interval
            if (e + 1) % self.config["training"]["log_interval"] == 0:
                self.logger.info(f'Epoch: {e + 1}; Train Loss: {round(train_loss, 3)}; Train mAP {round(trainmAP, 3)};')
            self.loss_meter["train"].reset()
            self.ap_meter["train"].reset()
            
            # checkpoint interval
            if (e + 1) % self.config["training"]["ckpt_interval"] == 0:
                self.val_meter = self.loop("validation")
                val_loss = self.loss_meter["validation"].average()
                val_AP = self.ap_meter["validation"].average()
                self.loss_meter["validation"].check(highest=False)
                self.save(e, self.ap_meter["validation"].check(highest=True) == 1)
                self.logger.info(f'Epoch: {e + 1}; Validation Loss: {round(val_loss, 3)}; Validation mAP {round(val_AP, 3)};')
                self.loss_meter["validation"].reset()
                self.ap_meter["validation"].reset()


        # save checkpoint at the end of training
        self.save(e, False)

        # output best metrics
        checkpoint = torch.load(self.model_dir / f"{self.name}_{self.lr}_best.pth")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info(f"Best validation AP \n {pprint.pformat(checkpoint['validation_AP'])}")
        self.logger.info(f"Best validation mAP {checkpoint['best_mAP']}")

        self.logger.info("Started testing best model!")

        self.test_meter = self.loop("test")
        test_results = {
            'test_AP': self.test_meter.value(self.sample_opposite, return_mean=False)[0],
            'test_loss': self.loss_meter["test"].average(),
            'test_mAP': self.ap_meter["test"].average(),
            'val_AP': checkpoint["validation_AP"],
            'val_mAP': checkpoint['best_mAP']
        }

        self.logger.info("Finished testing best model!")
        self.masterlogger.close()
        return test_results


    def save(self, epoch, best):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.loss_meter["validation"].highest,
            'best_mAP': self.ap_meter["validation"].highest,
            'validation_mAP': self.ap_meter["validation"].avg_list, # list of mAPs for each checkpoint interval
            'validation_AP': self.val_meter.value(self.sample_opposite, return_mean=False)[0],
            'train_loss': self.loss_meter["train"].avg_list,
            'config': self.config
        }
        torch.save(save_dict, self.model_dir / f"{self.name}_{self.lr}_checkpoint.pth")
        if best:
            torch.save(save_dict, self.model_dir / f"{self.name}_{self.lr}_best.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training hyper-parameters')
    parser.add_argument("--model_path", type=Path, required=True, help="Path to model")
    parser.add_argument('--data_dir', type=Path, required=True, help="Path to Epic-States dataset")
    parser.add_argument('--percent', type=float, default=100, help="Percentage of data to train on (Ex. 75)")
    parser.add_argument('--start_seed', dest='start_seed', type=int, default=0)
    parser.add_argument('--runs', dest='runs', type=int, default=5)
    parser.add_argument('--ftune', action='store_true')
    parser.add_argument('--log', type=Path)
    parser.add_argument('--use_head', action='store_true')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--sample_opposite', action='store_true')
    parser.add_argument('--ensemble_res', action='store_true')
    parser.add_argument('--novel', action='store_true')
    parser.add_argument('--gpu', default=0, type=int, help="GPU to use")
    args = parser.parse_args()

    args.config = Path("./configs/base.yaml") if not args.novel else Path("./configs/novel.yaml")
    aug_str, ensemble_str, head_str = "", "", ""
    if args.use_aug:
        print("Using Augmentation")
        aug_str = "-aug"
    if args.use_head:
        print("Using Head")
        head_str = "-head"
    if args.ensemble_res:
        print("Performing Ensemble")
        ensemble_str = "-ensemb"
    
    if str(args.model_path) == "resnet":
        pretraining_name = str(args.model_path)
    else:
        assert ".pth" in str(args.model_path)
        parent = args.model_path.parent
        pretraining_name = f"{parent.parent.stem}_{parent.stem}_{args.model_path.stem.split('checkpoint_')[-1]}"
    args.name = f"{pretraining_name}{ensemble_str}{head_str}{aug_str}_{args.percent}percent_{args.sample_opposite}sampled_ftune{args.ftune}"

    if args.novel and 'novel' not in str(args.log):
        args.log = args.log.parent / (args.log.stem + "_novel")
    args.model_dir = Path(args.log) / pretraining_name / "models"
    log_dir = args.log / pretraining_name / 'logs'
    results_dir = args.log / pretraining_name / 'results'
    args.model_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    best_metric = 0
    best_final = None
    seeds = list(range(args.start_seed, args.start_seed + args.runs))
    lrs = [1e-3]
    for lr in lrs:
        results = []
        results_names = []
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        for seed in seeds:
            experiment = Experiment(log_dir, args, seed, lr)
            out = experiment.run()
            results.append(out)
            results_names.append(experiment.name)
            
        with open(f"{results_dir / args.name}_finalresults_{lr}.pkl", 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Gather statistics over all seeds
        val_APs = [r['val_AP'] for r in results]
        test_APs = [r['test_AP'] for r in results]
        val_mAPs = np.array([r['val_mAP'] for r in results])
        best_seed = np.argmin(np.abs(val_mAPs - val_mAPs.mean()))
        best_name = results_names[best_seed]
        final = {
            'val_AP_classwise_mean': {cls: np.mean([dic[cls] for dic in val_APs]) for cls in val_APs[0].keys()},
            'val_AP_classwise_std': {cls: np.std([dic[cls] for dic in val_APs]) for cls in val_APs[0].keys()},
            'val_mAP': np.mean([r['val_mAP'] for r in results]),
            'val_mAP_std': np.std([r['val_mAP'] for r in results]),
            'test_AP_classwise_mean': {cls: np.mean([dic[cls] for dic in test_APs]) for cls in test_APs[0].keys()},
            'test_AP_classwise_std': {cls: np.std([dic[cls] for dic in test_APs]) for cls in test_APs[0].keys()},
            'test_mAP': np.mean([r['test_mAP'] for r in results]),
            'test_mAP_std': np.std([r['test_mAP'] for r in results]),
            'chosen_lr': lr,
            "chosen_seed": seeds[best_seed],
            "chosen_name": best_name,
        }
        
        # Keep only the models closest to the mean val mAP to save space
        # Deemed as the most representative training scheme
        for i,name in enumerate(results_names):
            if name == best_name:
                continue
            (args.model_dir / f"{name}_{lr}_checkpoint.pth").unlink()
            (args.model_dir / f"{name}_{lr}_best.pth").unlink()

        # print(json.dumps(final, indent=4, sort_keys=True))
        with open(f"{results_dir / args.name}_{lr}.json", 'w') as file:
            file.write(json.dumps(final, sort_keys=True, indent=4))
        
        if final["val_mAP"] > best_metric:
            best_metric = final["val_mAP"]
            best_final = final

    # Save the results from the best learning rate 
    with open(f"{results_dir / args.name}_final.json", 'w') as file:
         file.write(json.dumps(best_final, sort_keys=True, indent=4))


import os
import shutil
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse
from datetime import date
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import pdb
from html4vision import Col, imagetable
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler


from utils import load_config, Meter, Logger, APMeterCustom
from model import VOS, StateClassifier, TCN, HeadStateClassifier, FeatureExtractor, ActionClassifier, FeatureExtractorBN
from model import StateClassifierwithResNet, HeadStateClassifierwithResNet
from evaluation.data import StateDataLoader
from evaluation.evaluate import Experiment



class Visualization:
    def __init__(self, model_path, data_dir, config_path, gpu):
        self.model_path = model_path
        assert model_path.exists() and ".pth" in str(model_path)
        self.model_dict = torch.load(self.model_path)

        pretraining_name = model_path.parent.parent.stem
        self.output_dir = self.model_path.parent.parent / "vis" / self.model_path.stem.split(f"{pretraining_name}_")[-1] 
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.data_dir = data_dir
        self.config = self.model_dict["config"]
        self.masterlogger = Logger("main", str(self.output_dir / f"stdout.log"), mode='w')
        self.logger = self.masterlogger.logger

        self.use_head = "-head" in self.model_path.stem
        self.use_ensemble = "-ensemb" in self.model_path.stem
        self.use_ftune = "-ftune" in self.model_path.stem
        self.percent = self.config["training"]["percent"]
        self.sample_opposite = self.config['training']['sample_opposite']
        self.seed = int(self.model_path.stem.split("seed")[-1].split("_")[0])

        self.device = torch.device(gpu)
        batch_size = self.config["data"]["bs"]

        self.rng = np.random.default_rng(0)

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
        self.dic_opp = dic_opp

        classes = list(dic_opp.keys())
        classes.sort()
        self.classes = classes

        self.num2class = {i:j for i,j in enumerate(classes)}
        self.class2num = {j:i for i,j in enumerate(classes)}
        self.dic_num = {}
        for key in dic_opp.keys():
            k = self.class2num[key]
            v = [self.class2num[i] for i in dic_opp[key]]
            self.dic_num[k] = v

        # set up dataloaders
        self.datasets = {
            "train": StateDataLoader(self.logger, self.data_dir, "train", False, self.config["data"].get("filter", None), dic_opp,
                                        sample_opposite=self.sample_opposite,
                                        percent=self.percent,
                                        seed=self.seed),
            "validation": StateDataLoader(self.logger, self.data_dir, "validation", False, self.config["data"].get("filter", None), dic_opp),
            "test": StateDataLoader(self.logger, self.data_dir, "test", False, self.config["data"].get("filter", None), dic_opp)
        }

        self.data = {
            "train": DataLoader(self.datasets["train"],
                batch_size=batch_size, shuffle=False,
                num_workers=0,
                drop_last=False),
            "validation": DataLoader(self.datasets["validation"],
                batch_size=batch_size, shuffle=False,
                num_workers=0,
                drop_last=False),
            "test": DataLoader(self.datasets["test"],
                batch_size=batch_size, shuffle=False,
                num_workers=0,
                drop_last=False)
        }

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)

        if str(self.model_path) in ["resnet", "resnet50"]:
            use_resnet50 = str(self.model_path) == "resnet50"
            assert not (use_resnet50 and self.use_ensemble)
            model = StateClassifierwithResNet if self.use_ensemble else StateClassifier
            self.model = model(n_classes=len(classes), pretrained=True, ftune=False, resnet50=use_resnet50)
        else:
            model_info = torch.load(self.model_path, map_location='cuda:0')
            basemodel = FeatureExtractor(model_info["config"]["model"]["emb_size"])
            emb_size = model_info["config"]["model"]["emb_size"] if self.use_head else 512
            base_model = HeadStateClassifierwithResNet if self.use_ensemble else HeadStateClassifier
            self.model = base_model(trunk=basemodel.trunk, feat_size=emb_size, n_classes=len(self.classes), ftune=False)

        self.model.load_state_dict(self.model_dict["model_state_dict"])
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config["training"]["lr"], weight_decay=self.config["training"]["decay"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=10, threshold=1e-3, min_lr=1e-5, cooldown=5, verbose=True)
        self.loss_meter = {"train": Meter(), "validation": Meter(), "test": Meter()}
        self.ap_meter = {"train": Meter(), "validation": Meter(), "test": Meter()}
    

    def loop(self, split="train", num_batches=None):
        self.model.eval()
        self.model.set() # Set feature extractor in eval mode
        scores = []
        meter = APMeterCustom(self.dic_num, self.classes)
        for b, dic in enumerate(tqdm(self.data[split], desc=split, leave=False, dynamic_ncols=True)):
            dic = {key: value.to(self.device) for key, value in dic.items()}
            with torch.set_grad_enabled(False):
                out = self.model(dic)
                loss = Experiment.get_loss(out, dic, split, meter, self.percent)
                scores.append(out['probs'].detach().cpu().numpy())
            self.loss_meter[split].update(loss.item())
        self.ap_meter[split].update(meter.value(self.sample_opposite, return_mean=True))
        
        return meter, np.concatenate(scores, axis=0)

    def run(self, split):
        self.logger.info(f"Running visualization on {split}")
        self.meter, self.scores = self.loop(split)
        loss = self.loss_meter[split].average()
        ap = self.ap_meter[split].average()
        self.loss_meter[split].check(highest=False)
        self.loss_meter[split].reset()
        self.ap_meter[split].reset()

        APs, pr_curve = self.meter.value(use_opposite=self.sample_opposite, return_mean=False)
        self.save_pr_curves(pr_curve, split)
        # pdb.set_trace()
        class_thresholds = {cls:thr[np.argmax(2*pr*rec/(pr+rec-1e-8))] for cls, (pr, rec, thr) in pr_curve.items()}
        self.logger.info(f"Class thresholds: {class_thresholds}")
        state_estimates_df = self.datasets[split].vid_info.drop(["file", "path_mmap"], axis=1)
        state_estimates_df["pid"] = [Path(x).stem.split("_")[1] for x in state_estimates_df.path]
        state_estimates_df = state_estimates_df[["object", "pid", "path", "split", "state"]]

        state_estimates_df["state_estimates"] = self.get_states_from_scores(self.scores, class_thresholds, state_estimates_df.state.values)
        for i in range(len(self.classes)):
            state_estimates_df[f"score_{self.num2class[i]}"] = self.scores[:, i]
        
        csv_path = self.output_dir / f"{split}_state_estimates.csv"
        state_estimates_df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CSV to {csv_path}!")
        return state_estimates_df

    def get_states_from_scores(self, scores, class_thresholds, all_labels):
        state_estimates = []
        for b in range(scores.shape[0]):
            labels = all_labels[b].split(",")
            valid_classes = []
            for label in labels:
                valid_classes += [label] + self.dic_opp[label]
            states = {}
            for i in range(scores.shape[1]):
                class_str = self.num2class[i]
                if class_str not in valid_classes:
                    continue
                if scores[b, i] > class_thresholds[class_str]:
                    states[class_str] = scores[b, i]
            
            # choose maximum over opposite classes
            for cls in self.dic_opp:
                if cls not in states:
                    continue
                for opp in self.dic_opp[cls]:
                    if opp in states and states[cls] > states[opp]:
                        del states[opp]
            
            state_estimates.append(",".join(sorted(states.keys())))
        return state_estimates

    def save_pr_curves(self, pr_curve: dict, split: str):
        output_dir = self.output_dir / f"{split}_pr_curves"
        output_dir.mkdir(exist_ok=True, parents=True)
        for cls, (precisions, recalls, thresholds) in pr_curve.items():
            output_path = output_dir / f"{cls}.png"
            plt.figure()
            plt.plot(recalls, precisions)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR for {cls}")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR for all States")
        plt.set_cmap("viridis")
        for cls, (precisions, recalls, thresholds) in pr_curve.items():
            ax.plot(recalls, precisions, label=cls)
        
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(output_dir / "all.png", bbox_inches='tight')
        plt.close()

    def save_html(self, state_estimates: pd.DataFrame, num_images=100):
        html_root_path = self.output_dir / "state_estimations"
        html_root_path.mkdir(exist_ok=True, parents=True)
        images_dir = html_root_path / "images"
        images_dir.mkdir(exist_ok=True)
        state_estimates["new_path"] = [str(images_dir / Path(x).name) for x in state_estimates.path.values]
        for src, dest in zip(state_estimates.path.values, state_estimates.new_path.values): 
            shutil.copy(src, dest)
        
        state_estimates["relative_path"] = [str(Path(x).relative_to(html_root_path)) for x in state_estimates.new_path.values]
        
        for state in ['all'] + list(self.dic_opp.keys()):
            df_state = state_estimates
            if state != 'all':
                df_state = state_estimates[[state in x.split(", ") for x in state_estimates.state.values]]
                df_state = df_state.sort_values(by=[f"score_{self.dic_opp[state][0]}"], ascending=False).iloc[:num_images]
            else:
                df_state = df_state.sample(n=min(num_images, df_state.shape[0]), random_state=0).copy()
            html_path = html_root_path / f"{state}.html"
            cols = [
                Col('id1', 'ID'),
                Col('img', 'Image', df_state.relative_path.values.tolist()),
                Col('text', 'Ground Truth', df_state.state.values.tolist()),
                Col('text', 'Estimate', df_state.state_estimates.values.tolist()),
            ]

            if state == 'all':
                cols += [Col("text", f"score_{cls}", [f"{x:0.3f}" for x in df_state[f"score_{cls}"].values]) for cls in self.classes]
            else:
                cols += [Col("text", f"score_{cls}", [f"{x:0.3f}" for x in df_state[f"score_{cls}"].values]) for cls in [state, self.dic_opp[state][0]]]

            imagetable(cols, html_path, f'{state} Predictions',
                        imscale=1.0,
                        sortcol=None,
                        sortable=True,
                        sticky_header=True,
                        sort_style='materialize',
                        zebra=True,
            )    

            self.logger.info(f"Saved HTML to {html_path}!")

    def generate_participant_wise_eval(self, split):
        csv_path = self.output_dir / f"{split}_state_estimates.csv"
        state_estimates = pd.read_csv(csv_path)
        pids = set(state_estimates.pid.values)
        maps = {}
        print()
        print(split)
        for pid in sorted(pids):
            df = state_estimates[state_estimates.pid == pid]
            labels = [i.split(',') for i in df.state]
            mlb = MultiLabelBinarizer(classes=self.classes)
            vectorised_labels = mlb.fit_transform(labels)
            probs = df[[f"score_{cls}" for cls in self.classes]].values
            meter = APMeterCustom(self.dic_num, self.classes)
            meter.add(probs, vectorised_labels)
            print(pid)
            print(meter.value(return_mean=False)[0])
            print()
            maps[pid] = [
                meter.value(return_mean=True, weighted=False), 
                meter.value(return_mean=True, weighted=True), 
            ]
            # maps[f"{pid}_numlab"] = sum([len(lab) for lab in labels])
        csv_path = self.output_dir / f"{split}_participant_avgaps.csv"
        maps["name"] = [
            self.model_path.stem.split("_best")[0],
            "weighted_" + self.model_path.stem.split("_best")[0]
        ]
        maps = pd.DataFrame.from_dict(maps)
        maps.set_index("name", inplace=True)
        maps.to_csv(csv_path, index=True)
        print(f"Saved participant-wise evaluation to {csv_path}")

    def generate_aps_per_class(self, split):
        csv_path = self.output_dir / f"{split}_state_estimates.csv"
        state_estimates = pd.read_csv(csv_path)
        df = state_estimates
        labels = [i.split(',') for i in df.state]
        mlb = MultiLabelBinarizer(classes=self.classes)
        vectorised_labels = mlb.fit_transform(labels)
        probs = df[[f"score_{cls}" for cls in self.classes]].values
        meter = APMeterCustom(self.dic_num, self.classes)
        meter.add(probs, vectorised_labels)
        aps, _ = meter.value(return_mean=False)
        aps["name"] = self.model_path.stem.split("_best")[0]
        aps = {k:[v] for k,v in aps.items()}
        csv_path = self.output_dir / f"{split}_classwise_aps.csv"
        df_aps = pd.DataFrame.from_dict(aps)
        df_aps.set_index("name", inplace=True)
        df_aps.to_csv(csv_path, index=True)
        print(f"Saved classwise AP evaluation to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training hyper-parameters')
    parser.add_argument('--data_dir', "-d", type=Path, default="/home/mohit/VOS/EPIC-2018/data_dir")
    parser.add_argument('--config', dest='config', type=str, default="./configs/base_final.yaml")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to models")
    parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU to use")
    args = parser.parse_args()

    vis = Visualization(args.model_path, args.data_dir, args.config, args.gpu)
    for split in ["validation", "test"]:
        # state_estimates = vis.run(split)
        # vis.save_html(state_estimates)
        vis.generate_participant_wise_eval(split)
        # vis.generate_aps_per_class(split)

    


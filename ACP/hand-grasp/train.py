import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from helper import set_numpythreads, str2bool
set_numpythreads()
import argparse
import pprint
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils import load_config, Meter, Logger, APMeterCustom
from models import ClassificationNet, SimCLRwGUN71
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from losses import NTXentLoss

class Experiment:
    def __init__(self, config_path, logdir, name, model_dir, args):
        self.name = name
        self.model_dir = model_dir
        self.logdir = logdir
        self.masterlogger = Logger("main", os.path.join(self.logdir, f"{self.name}.log"))
        self.logger = self.masterlogger.logger
        self.config = load_config(config_path)
        self.tsc = args.tsc


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
        self.logger.info(f"TSC set to {self.tsc}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_cfmt = args.save_cfmt

        self.rng = np.random.default_rng(0)

        self.get_dataloader()

        # Get class info and weights from the dataloader
        self.classes = self.datasets["train"].classes
        self.weights = {key: torch.from_numpy(self.datasets[key].weights).type(torch.float32).to(self.device) for key in self.datasets.keys()}

        if not args.tsc:
            self.model = ClassificationNet(n_classes=len(self.classes)).to(self.device)
        else:
            batch_size = self.config["data"]["bs"]
            self.model = SimCLRwGUN71(nclasses=len(self.classes), model_config=self.config['model']).to(self.device)
            self.criterion = NTXentLoss(self.device, batch_size, self.config["model"]["temperature"], True)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config["training"]["lr"], weight_decay=self.config["training"]["decay"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=5, threshold=1e-4, min_lr=1e-5, cooldown=5, verbose=True)
        self.loss_meter = {"train": Meter(), "validation": Meter(), "test": Meter()}
        self.ap_meter = {"train": Meter(), "validation": Meter(), "test": Meter()}
        self.acc_meter = {"train": Meter(), "validation": Meter(), "test": Meter()}
        self.start_epoch = 0

        # try to restore checkpoint
        if self.config["training"].get("resume", True):
            try:
                checkpoint = torch.load(os.path.join(self.model_dir, f"{self.name}_checkpoint.pth"))
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.loss_meter["validation"].lowest = checkpoint["best_loss"]
                self.ap_meter["validation"].highest = checkpoint["best_mAP"]
                self.acc_meter["validation"].highest = checkpoint["best_acc"]
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

        os.makedirs(os.path.join(self.logdir, self.name), exist_ok=True)
        os.makedirs(os.path.join(self.logdir, self.name + "_results"), exist_ok=True)
        self.train_writer = SummaryWriter(os.path.join(self.logdir, self.name))
        
    def get_dataloader(self):
        global worker_init_fn
        if self.tsc:
            from data.SimCLR import EpicHandTrackswGUN71, worker_init_fn
            self.datasets = {
                "train": EpicHandTrackswGUN71(self.config, "train", True),
                "validation": EpicHandTrackswGUN71(self.config, "val", False),
                # Use validation for test as well
                "test": EpicHandTrackswGUN71(self.config, "val", False),
            }
            
        else:
            from data.GUN71 import HandDataLoader, worker_init_fn
            self.datasets = {
                "train": HandDataLoader(self.config, "train", True),
                "validation": HandDataLoader(self.config, "val", False),
                "test": HandDataLoader(self.config, "test", False),
            }

        # set up dataloaders
        self.data = {
            "train": DataLoader(self.datasets["train"],
                batch_size=self.config["data"]["bs"], shuffle=False,
                num_workers=self.config["data"]["nw"],
                drop_last=False),
            "validation": DataLoader(self.datasets["validation"],
                batch_size=self.config["data"]["bs"], shuffle=False,
                num_workers=self.config["data"]["nw"],
                drop_last=False),
            "test": DataLoader(self.datasets["test"],
                batch_size=self.config["data"]["bs"], shuffle=False,
                num_workers=self.config["data"]["nw"],
                drop_last=False)
        }

        return None
        
    def save(self, epoch, best):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.loss_meter["validation"].lowest,
            'best_mAP': self.ap_meter["validation"].highest,
            'best_acc': self.acc_meter["validation"].highest,
            'validation_loss': self.loss_meter["validation"].avg_list,
            'validation_AP': self.val_meter.value(return_mean=False),
            'validation_mAP': self.val_meter.value(return_mean=True),
            'validation_acc': self.val_meter.get_acc(),
            'train_loss': self.loss_meter["train"].avg_list,
            'config': self.config
        }
        torch.save(save_dict, os.path.join(self.model_dir, f"{self.name}_checkpoint.pth"))

        if self.config["training"].get("save_after", None) is not None:
            if (epoch + 1) % (self.config["training"].get("save_after", None) * self.config["training"]["ckpt_interval"]) == 0:
                torch.save(save_dict, os.path.join(self.model_dir, f"{self.name}_checkpoint_{epoch + 1}.pth"))                
        if best:
            torch.save(save_dict, os.path.join(self.model_dir, f"{self.name}_best.pth"))

    def load_best(self):
        try:
            checkpoint = torch.load(os.path.join(self.model_dir, f"{self.name}_best.pth"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded Best Model")
        except OSError as e:
            self.logger.error(e)
            self.logger.info("Best Model Not Saved")

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
                worker_init_fn=worker_init_fn,
                pin_memory=True,
                drop_last=False)

    def get_loss(self, dic, split, meter):

        logits = self.model.forward_classifier(dic['img'])

        lab = dic['label']

        loss = F.cross_entropy(logits, lab, weight=self.weights[split])
        meter.add(F.softmax(logits, dim=1).detach().cpu().numpy(), lab.detach().cpu().numpy())

        return loss
    
    def get_loss_simclr(self, dic, split, meter):

        logits = self.model.forward_classifier(dic['img'])

        emb1 = self.model.forward_embedding(dic["f1"])
        emb2 = self.model.forward_embedding(dic["f2"])
        loss_simclr = self.criterion(emb1, emb2)

        lab = dic['label']

        loss_classification = F.cross_entropy(logits, lab, weight=self.weights[split])

        loss = self.config['training']['class_w'] * loss_classification + self.config['training']['tsc_w'] * loss_simclr
        meter.add(F.softmax(logits, dim=1).detach().cpu().numpy(), lab.detach().cpu().numpy())

        return loss

    def loop(self, split="train", num_batches=None):
        if 'train' in split:
            self.model.train()
            if num_batches is not None:
                self.create_train_dataloader(num_batches)
        else:
            self.model.eval()

        meter = APMeterCustom(classes=self.classes)
        for b, dic in enumerate(tqdm(self.data[split])):
            dic = {key: value.cuda() for key, value in dic.items()}
            with torch.set_grad_enabled('train' in split):
                if not self.tsc:
                    loss = self.get_loss(dic, split, meter)
                else:
                    loss = self.get_loss_simclr(dic, split, meter)
                # backpropagate if training
                if 'train' in split:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            self.loss_meter[split].update(loss.item())
        self.ap_meter[split].update(meter.value(return_mean=True))
        self.acc_meter[split].update(meter.get_acc())
        
        return meter

    def add_dic_to_logs(self, dic, writer, split, epoch):
        for key, val in dic.items():
            writer.add_scalar(f"{split}_{key}", val, epoch)

    def run(self):
        best_before_epochs = 0
        for e in range(self.start_epoch, self.start_epoch + self.config["training"]["num_epochs"]):

            # run training loop
            self.train_meter = self.loop("train", self.config["training"]["num_batches"])
            train_loss = self.loss_meter["train"].average()
            self.scheduler.step(train_loss)
            
            # log interval
            if (e + 1) % self.config["training"]["log_interval"] == 0:
                trainmAP = self.ap_meter["train"].average()
                trainacc = self.acc_meter["train"].average()
                self.logger.info(f'Epoch: {e + 1}; Train Loss: {round(train_loss, 3)}; Train mAP {round(trainmAP, 3)}; Train Acc {round(trainacc, 3)}')
                self.train_writer.add_scalar('training_loss', train_loss, e)
                self.train_writer.add_scalar('training_mAP', trainmAP, e)
                self.train_writer.add_scalar('training_acc', trainacc, e)
                self.loss_meter["train"].reset()
                self.ap_meter["train"].reset()
                self.acc_meter["train"].reset()

                # Add AP per class to tensorboard
                APs = self.train_meter.value()
                self.add_dic_to_logs(APs, self.train_writer, "training", e)

            # checkpoint interval
            if (e + 1) % self.config["training"]["ckpt_interval"] == 0:
                self.val_meter = self.loop("validation")
                val_loss = self.loss_meter["validation"].average()
                val_AP = self.ap_meter["validation"].average()
                valacc = self.acc_meter["validation"].average()
                self.acc_meter["validation"].check(highest=True) # Update best Acc
                self.loss_meter["validation"].check() # Update best loss
                isbetter = self.ap_meter["validation"].check(highest=True) == 1
                best_before_epochs = 0 if isbetter else best_before_epochs + self.config["training"]["ckpt_interval"]
                self.save(e, isbetter)
                self.logger.info(f'Epoch: {e + 1}; Validation Loss: {round(val_loss, 3)}; Validation mAP {round(val_AP, 3)}; Validation Acc {round(valacc, 3)};')
                self.train_writer.add_scalar('validation_loss', val_loss, e)
                self.train_writer.add_scalar('validation_mAP', val_AP, e)
                self.train_writer.add_scalar('validation_acc', valacc, e)
                self.loss_meter["validation"].reset()
                self.ap_meter["validation"].reset()
                self.acc_meter["validation"].reset()

                # Add AP per class to tensorboard
                APs = self.val_meter.value()
                self.add_dic_to_logs(APs, self.train_writer, "validation", e)

                if best_before_epochs > self.config["training"]["max_epochs_before_impr"]:
                    self.logger.info(f"Validation metric has not improved for {self.config['training']['max_epochs_before_impr']} epochs. Quitting Training...")
                    break

        # save checkpoint at the end of training
        self.save(e, False)

        # output best metrics
        checkpoint = torch.load(os.path.join(self.model_dir, f"{self.name}_best.pth"))
        self.logger.info("Best validation AP \n {}".format(pprint.pformat(checkpoint['validation_AP'])))
        self.logger.info("Best validation mAP {}".format(checkpoint['validation_mAP']))
        self.logger.info("Best validation acc {}".format(checkpoint['validation_acc']))

        self.load_best()

        self.logger.info("Started Testing best model!")        

        self.test_meter = self.loop("test")
        test_results = {}
        test_results['test_AP'] = self.test_meter.value()
        test_results['test_loss'] = self.loss_meter["test"].average()
        test_results['test_mAP'] = self.ap_meter["test"].average()
        test_results['test_acc'] = self.acc_meter["test"].average()
        test_results['val_mAP'] = checkpoint['validation_mAP']

        print("Finished Testing")

        cm = self.test_meter.get_cm(normalize=None)
        cm_frame = pd.DataFrame(cm, index=self.classes, columns=self.classes)
        cm_frame.to_csv(f"{os.path.join(self.logdir, self.name + '_results')}/cm.csv")
        print("Finished saving raw Confusion Matrix")
        if self.save_cfmt:
            self.test_meter.plot_cm(normalize="true", path=f"{os.path.join(self.logdir, self.name + '_results')}/cm.pdf")
            print("Finished creating Confusion Matrix Plot")
        self.test_meter.to_pickle(path=f"{os.path.join(self.logdir, self.name + '_results')}/test_meter.pkl")
        print("Finished saving test_meter")

        self.logger.info("Test AP \n {}".format(pprint.pformat(test_results['test_AP'])))
        self.logger.info("Test mAP {}".format(test_results['test_mAP']))
        self.logger.info("Test acc {}".format(test_results['test_acc']))


        self.masterlogger.close()

        return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training hyper-parameters')
    parser.add_argument('--config', dest='config', type=str,
                        default="hand-grasp/configs/base_classifier.yaml")
    parser.add_argument('--save_cfmt', dest='save_cfmt', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--modeldir', dest='modeldir', type=str,
                        default="./models/GUN-71")
    parser.add_argument('--log', dest='log', type=str,
                        default="./logs/GUN-71")
    parser.add_argument('--name', dest='name', type=str, default="")
    parser.add_argument('--seed', dest='seed', type=int,
                        default=0)
    parser.add_argument('--tsc', dest='tsc', action='store_true', default=False)
    

    args = parser.parse_args()

    os.system("mkdir -p {}".format(args.modeldir))
    os.system("mkdir -p {}".format(args.log))

    if args.tsc:
        args.name += "_tsc"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    
    args.name = args.name + f"_seed{args.seed}"

    exp = Experiment(args.config, args.log, args.name, args.modeldir, args)
    exp.rng = np.random.default_rng(args.seed)
    out = exp.run()

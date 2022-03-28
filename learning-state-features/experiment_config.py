from abc import ABC
import argparse
import shutil
import pdb
import pprint
import torch
from datetime import date
from pathlib import Path
import yaml
import glob

import torch.optim as optim
from losses import NTXentLoss
from model import SimCLRLayer
from torch.utils.tensorboard import SummaryWriter

from utils import Logger, Meter, init_logger, load_config, global_logger, set_torch_seed

torch.set_num_threads(2)

def build_common_train_parser(config_path):
    parser = argparse.ArgumentParser(description='training hyper-parameters')
    parser.add_argument('--config', type=Path, default=config_path, help="Path to config file.")
    parser.add_argument('--model', type=Path, default='./models', help="Path to save output models")
    parser.add_argument('--gpu', '-g', type=int, default=0, help="GPU to use")
    parser.add_argument('--name', "-n", type=str, default="temp", help="Name of experiment")
    parser.add_argument('--seed', "-s", type=int, default=0)
    return parser

class ExperimentConfig(ABC):
    def __init__(self, args: argparse.Namespace, initialize_model=True, **kwargs):

        config_path = args.config
        model_dir = args.model
        gpu = args.gpu
        seed = args.seed
        name = args.name

        self.config = load_config(config_path)
        self.config["args"] = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
        self.data_suffix = self.config["data"].get("suffix", None)
        if 'data_suffix' in kwargs:
            self.data_suffix = kwargs['data_suffix']
        assert self.data_suffix is not None

        self.name = date.today().strftime("%Y-%m-%d") + f"_{name}" + f"_sd{seed}"
        self.model_dir = model_dir / self.data_suffix / self.name
        self.model_dir.mkdir(exist_ok=True, parents=True)

        assert not (self.model_dir / "checkpoint_200.pth").exists(), "Directory has trained model already!"

        with open(self.model_dir / f"config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self.device = torch.device(gpu)
        self.min_segment_len = self.config["data"].get("minlen", 0)
        self.max_segment_len = self.config["data"].get("maxlen", 0)
        self.batch_size = self.config["data"]["bs"]
        self.segment_range = (self.min_segment_len, self.max_segment_len)

        self.logger = init_logger(self.model_dir / f"stdout.log")
        self.logger.info(f"Using GPU {gpu}: {torch.cuda.get_device_name(gpu)}")
        self.logger.info(pprint.pformat(self.config))
        self.logger.info(f"Saving outputs to {self.model_dir}")
        self.logger.info(f"Using datasets: (train, val)_{self.data_suffix}")

        set_torch_seed(seed)
        self.logger.info(f"Set seed to {seed}")

        if initialize_model:
            self.model = SimCLRLayer(512, 128).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["training"]["lr"])
        
        self.loss_meter = {"train": Meter(), "validation": Meter()}

        self.start_epoch = 0

        # try to restore checkpoint
        if initialize_model and self.config["training"].get("resume", False):
            self.resume()
            self.logger.info(f"Loaded model. Resuming training from epoch {self.start_epoch}")
        elif not initialize_model:
            self.logger.info(f"Training model from scratch")
            for f in glob.glob(str(self.model_dir) + "/events.out.tfevents.*"):
                path = Path(f)
                self.logger.info(f"Removing stale TensorBoard file: {path}")
                path.unlink()
        
        self.tb_writer = SummaryWriter(self.model_dir)
        

    def resume(self):
        model_name = self.config["training"].get("resume_name", f"checkpoint_last.pth")
        checkpoint = torch.load(self.model_dir / model_name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_meter["validation"].lowest = checkpoint["best_loss"]
        self.loss_meter["train"].avg_list = checkpoint["train_loss"]
        self.loss_meter["validation"].avg_list = checkpoint["validation_loss"]
        self.config = checkpoint["config"]
        self.start_epoch = checkpoint["epoch"] + 1

    def load_base_model(self, model_path):
        model_info = torch.load(model_path, map_location=self.device)
        model_weights = {k: model_info["model_state_dict"]["network." + k]
                         for k in self.model.network.state_dict().keys()}
        self.model.network.load_state_dict(model_weights)
        self.logger.info(f"Loaded base model at {model_path}")

    def save(self, epoch, best):
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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

    
    def run(self):
        for e in range(self.start_epoch, self.config["training"]["num_epochs"]):
            # run training loop
            self.current_epoch = e
            self.loop("train", self.config["training"]["num_batches"])
            
            # log interval
            if (e + 1) % self.config["training"]["log_interval"] == 0:
                train_loss = self.loss_meter["train"].average()
                self.tb_writer.add_scalar("loss/train", train_loss, e)
                self.loss_meter["train"].reset()
                train_acc = None
                if hasattr(self, 'acc_meter'):
                    train_acc = self.acc_meter['train'].average()
                    self.acc_meter["validation"].reset()
                    self.tb_writer.add_scalar("acc/train", train_acc, e)
                self.logger.info(self.construct_logstr('train', e, train_loss, train_acc))
            
            # checkpoint interval
            if (e + 1) % self.config["training"]["ckpt_interval"] == 0:
                self.loop("validation")
                val_loss = self.loss_meter["validation"].average()
                self.tb_writer.add_scalar("loss/val", val_loss, e)
                self.save(e, self.loss_meter["validation"].check() == -1)
                val_acc = None
                if hasattr(self, 'acc_meter'):
                    val_acc = self.acc_meter["validation"].average()
                    self.acc_meter["validation"].reset()
                    self.tb_writer.add_scalar("acc/val", val_acc, e)
                self.logger.info(self.construct_logstr("validation", e, val_loss, val_acc))
                self.loss_meter["validation"].reset()
                
        
        # save checkpoint at the end of training
        self.save(e, False)

    
    def construct_logstr(self, split, e, loss, acc=None):
        Split = split.capitalize()
        loss_str = f"{Split + ' Loss:':<16} {loss:0.6f}"
        acc_str = f"{Split + ' Accuracy:':<20} {acc*100:06.3f}%" if acc is not None else ""
        log = f"Epoch: {e}; {loss_str}; {acc_str}"
        return log
    
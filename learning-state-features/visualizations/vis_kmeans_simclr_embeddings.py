import sys, os, argparse, pdb
from pathlib import Path
import pickle as pkl
import copy
import torch
import torch.nn as nn
import torch.utils.data as tdh
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import numpy as np
from scipy import interpolate

import pprint
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageDraw

import sklearn.cluster

import argparse
from datetime import date
import pdb
import pprint
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from html4vision import Col, imagetable
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader, Dataset

from utils import load_config, Meter, Logger, set_torch_seed, worker_init_fn, get_worker_generator
from model import SimCLRLayer, Resnet
from data.transforms import unnormalize

from matplotlib import cm


def get_image_from_array(fs):
	fsarr = fs.repeat(128).reshape(1, 128 * len(fs)).repeat(16, axis=0)
	fsimg = Image.fromarray(np.uint8(cm.inferno(fsarr)*255))
	return fsimg

		
def cluster_affinity(cluster_id, assignment, scores, files, indices, nsamples, out_dir, model_name, split, save=False):
	# get pertinent frames
	track_idxs = np.where(assignment == cluster_id)
	track_files = [files[i] for i in track_idxs[0]]
	selected_indices = indices[track_idxs[0]]
	selected_scores = scores[track_idxs[0]]

	nsamples = min(nsamples, selected_indices.shape[0])
	for i, ind in enumerate(np.argsort(selected_scores)[:nsamples]):
		frame = extract_frame(track_files[ind], selected_indices[ind].item())
		if save:
			frame.save(f"{os.path.join(out_dir, 'clusters')}/{model_name}/{split}/{cluster_id}_{str(i).zfill(3)}.jpg")



def main(args):
	base_data_dir = args.data
	model_name = args.model_name
	model_path = f"/home/mohit/VOS/video-hand-grasps/models/EPIC-KITCHENS/{model_name}_best.pth"
	out_dir = args.out_dir

	# load model
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	best_model = torch.load(model_path)
	config = best_model["config"]
	model = SimCLR(config['model']).to(device)
	model.load_state_dict(best_model["model_state_dict"])
	model.eval()

	# peripherals
	global right_only
	right_only = config["data"].get("right_only", False)

	global num_clusters
	num_clusters = 64

	print(f"Right only: {right_only}")

	split = args.split
	# set up dataloader
	dataset = EpicHandTracks4TCNVis(base_data_dir, split=split,
									subset=config["data"].get("filter", None),
									imsize=config["data"]["imsize"],
									min_track_len=config["data"].get("min_track_len", 10),
									right_only=config["data"].get("right_only", False),
									window=config["data"].get("window", 10),
									margin=config["data"].get("margin", 4),
									transform=False
								 )
	dataloader = tdh.DataLoader(dataset,
								batch_size=config["data"]["bs"], shuffle=False,
								num_workers=config["data"]["nw"])

	assignment, indices, scores = get_scores(dataloader, model, device)

	print(assignment.shape, indices.shape, scores.shape)

	cluster_ids, cluster_size = np.unique(assignment, return_counts=True)
	print(f"Num Active Clusters: {cluster_ids.shape[0]}")
	cluster_sizes = {cluster_ids[i]: cluster_size[i] for i in range(cluster_ids.shape[0])}
	print(cluster_sizes)

	n = 3
	repeated_folders = [item for item in dataset.track_folders for i in range(n)]

	for i in tqdm(range(num_clusters)):
		cluster_affinity(i, assignment[::3], scores[::3], dataset.track_folders, indices[::3], 200, out_dir, model_name, split, save=True)


def image_grid(imgs, rows, cols):
    assert len(imgs) <= rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


class EK55_ObjectDataset(Dataset):
	def __init__(self, data_dir, obj) -> None:
		super(EK55_ObjectDataset, self).__init__()
		self.data_dir = Path(data_dir)
		obj_names = list(os.listdir(self.data_dir))
		self.images = []
		self.labels = []
		for name in obj_names:
			if name != obj:
				continue
			images = [self.data_dir / name / f for f in os.listdir(self.data_dir / name)]
			self.images += images
			self.labels += [name] * len(images)

		self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img = Image.open(self.images[index])
		return self.transform(img), self.labels[index], str(self.images[index])


class Visulization:
	def __init__(self, data_dir: Path, model_path: Path, vis_root_dir: Path, num_clusters: int, object: str):
		self.name = model_path.stem
		self.num_clusters = num_clusters
		self.model_path = model_path
		self.data_dir = data_dir
		self.vis_dir = vis_root_dir / self.name
		self.object = object
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		os.system(f"mkdir -p {self.vis_dir}")

		self.logger = Logger("main", self.vis_dir / "output.log").logger

		# set up dataloaders
		dataset = EK55_ObjectDataset(data_dir, self.object)

		g = get_worker_generator(seed=0)
		self.data = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True) 
		self.logger.info(f"Found {len(dataset)} images.")

		self.model = SimCLRLayer(512, 128).to(self.device) if self.name != "resnet" else Resnet(512).to(self.device)

		if self.name != "resnet":
			checkpoint = torch.load(self.model_path)
			self.model.load_state_dict(checkpoint["model_state_dict"])
			self.logger.info("Loaded model!")
		else:
			self.logger.info("Loaded pretrained resnet!")

	def inference(self):
		batch_count = 0
		self.model.eval()
		embeddings = []
		labels = []
		paths = []
		for b, data in enumerate(tqdm(self.data, leave=False, dynamic_ncols=True)):
			batch_count += 1
			images = data[0].to(self.device)
			with torch.set_grad_enabled(False):
				emb1 = self.model(images)
				embeddings.append(emb1)
				labels += data[1]
				paths += data[2]

		return torch.cat(embeddings, dim=0).cpu().numpy(), labels, paths


	def cluster(self, embeddings):
		kmeans = sklearn.cluster.KMeans(n_clusters=self.num_clusters, random_state=0).fit(embeddings)
		assignment = kmeans.labels_
		scores = -kmeans.transform(embeddings)[np.arange(len(assignment)), assignment]
		print(scores[0:10])
		return assignment, scores


	def run(self):
		embeddings, labels, paths = self.inference()
		assignments, scores = self.cluster(embeddings)
		vis_data = []
		for i in range(len(assignments)):
			vis_data.append({"cluster":assignments[i], "score":scores[i], "path":Path(paths[i]), "label":labels[i]})
		self.generate_html(vis_data)
		self.logger.info(f"Finished! Outputs located at {self.vis_dir}")


	def generate_html(self, vis_data):
		cluster_dir = self.vis_dir / "clusters" / self.object
		if cluster_dir.exists():
			os.system(f"rm -r {cluster_dir}")
		cluster_dir.mkdir(parents=True)

		clusters = {i:[] for i in range(self.num_clusters)}
		for ex in vis_data:
			a = ex["cluster"]
			clusters[a].append(ex)
		
		clusters_small = {i:[] for i in range(self.num_clusters)}
		for i in clusters:
			examples = np.random.choice(clusters[i], size=min(len(clusters[i]), 50), replace=False)
			clusters_small = examples
			images = []
			for ex in examples:
				images.append(Image.open(ex['path']))
			grid = image_grid(images, rows=5, cols=10)
			grid.save(cluster_dir / f"{i:02d}.jpg")
			

		# cols = [
		# 	Col('id1', 'ID'),
		# 	Col('img', 'Images', [x.relative_to(self.vis_dir) for x in vis_data["pos_paths"]]),
		# 	Col('img', 'Scores', [x.relative_to(self.vis_dir) for x in vis_data["neg_paths"]]),
		# 	Col('img', 'Best Negative', [x.relative_to(self.vis_dir) for x in vis_data["best_neg_paths"]]),
		# 	Col('text', 'Losses', [str([float(f"{y:.3f}") for y in x]) for x in vis_data["losses"]]),
		# 	Col('text', 'Best Loss', [float(f"{x:.3f}") for x in vis_data["best_losses"]]),
		# 	Col('text', 'Pos Sim', [float(f"{x:.3f}") for x in vis_data["pos_sim"]]),
		# 	Col('text', 'Neg Sim', [str([float(f"{y:.3f}") for y in x]) for x in vis_data["neg_sim"]]),
		# ]

		# imagetable(cols, Path(self.vis_dir) / 'multi_instance.html', 'Multi-instance Track Visualizations',
		# 			imscale=1.0,                # scale images to 0.4 of the original size
		# 			sortcol=5,                  # initially sort based on column 2 (class average performance)
		# 			sortable=True,              # enable interactive sorting
		# 			sticky_header=True,         # keep the header on the top
		# 			sort_style='materialize',   # use the theme "materialize" from jquery.tablesorter
		# 			zebra=True,                 # use zebra-striped table
		# )  


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='training hyper-parameters')
	parser.add_argument('--data', type=Path, default="/home/mohit/VOS/EPIC-2018/EK55_crops")
	parser.add_argument('--model_path', type=Path)
	parser.add_argument("--out_dir", type=Path, default="/data01/smodi9/VOS/vis/kmeans")
	parser.add_argument("--object", required=True, type=str, help="Object to cluster")
	parser.add_argument("--num_clusters", type=int, default=5, help="Number of clusters for KMeans")
	args = parser.parse_args()

	assert args.model_path.exists() or str(args.model_path) == "resnet"
	
	set_torch_seed(0)
	vis = Visulization(args.data, args.model_path, args.out_dir, args.num_clusters, args.object)
	vis.run()
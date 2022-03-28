import sys, os, torch, lzma
from scipy import ndimage
import numpy as np
import glob
from PIL import Image, ImageFilter
from sklearn import metrics
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
import time
import argparse
import pprint
import pandas as pd
import yaml, cv2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from visualize_utils import to_tensor, to_PIL, overlay_colored_heatmaps

def load_config(path):
	with open(path, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	return config

def plot_boundary(mask, img):
	contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Define sufficient enough colors for blobs
	colors = [(255, 255, 0), (40, 124, 48), (0, 255, 0), (255, 0, 0), (0, 255, 0)]

	# Draw all contours, and their children, with different colors
	out = img[:, :, ::-1].astype(np.uint8).copy()
	k = -1
	for i, cnt in enumerate(contours):
		if (hier[0, i, 3] == -1):
			k += 1
		cv2.drawContours(out, [cnt], -1, colors[0], 4)
	return out

import cycler
plt.rcParams['image.cmap'] = 'Set1'

class_info = load_config("./configs/class_info.yaml")
sel_classes = class_info['easy_classes']['classes']
sel_names = class_info['easy_classes']['names']

GT_GRASP_LIST = sel_classes

def read_file(fp):
	return pickle.load(lzma.open(fp, "rb"))

def fileparts(path):
	home = os.path.dirname(path)
	basename = os.path.basename(path)
	basenamenoext, ext = os.path.splitext(basename)

	return home, basename, basenamenoext, ext

def process_image(fname, name, path, sz):
	if "random" in name:
		# pred_GTactions = np.zeros((len(GT_GRASP_LIST), sz[1], sz[0])) + 0.5
		pred_GTactions = np.random.uniform(0, 1.0, (len(GT_GRASP_LIST), sz[1], sz[0]))
	else:
		pred_img = f"{path}/final_{fname[:-3]}npy"
		if sz is not None:
			pred = F.interpolate(torch.from_numpy(np.load(pred_img)).permute(2, 0, 1).unsqueeze(0), sz[::-1])[0]
			pred_GTactions = torch.zeros(len(GT_GRASP_LIST), sz[1], sz[0])
			for i in range(len(GT_GRASP_LIST)):
				pred_GTactions[i] += pred[i]
			pred_GTactions = pred_GTactions.numpy()
		else:
			pred_GTactions = np.load(pred_img)
	# pred_GTactions = ndimage.gaussian_filter(pred_GTactions, sigma=(0, 3, 3), order=0)
	return np.round(pred_GTactions * 255, 1)

def save_object(obj, filename):
	with open(filename, 'wb') as outp:  # Overwrites any existing file.
		pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def create_vis(img, hm):
	sel_indices = [0, 11]# [0, 2, 9, 27]
	colors = ["green", "blue", "magenta", 'red']

	imgtensor = to_tensor(img)
	hmtensor = torch.from_numpy(hm)[:, :, sel_indices].permute(2, 0, 1)

	overlay = overlay_colored_heatmaps(imgtensor, hmtensor, colors, True)

	ol = to_PIL(overlay)
	img = to_PIL(imgtensor)

	return img, ol

def main(args):
	h, b, bwoext, ext = fileparts(args.config)
	if b != "defaults_vis.yaml":
		args.out_dir = os.path.join(args.out_dir, bwoext)
		
	config = load_config(args.config)
	methods = config['methods']

	os.makedirs(args.out_dir, exist_ok=True)
	eval_imgsize = (300, 200)
	
	predictions = {}

	images = sorted(glob.glob(f"{args.imdir}/{args.split}_new/*.jpg"))
	GT_data = {}

	for img in images:
		GT = read_file(img[:-4] + "_proc.xz")
		if eval_imgsize is not None:
			GT['ord_masks'] = [Image.fromarray(m).resize(eval_imgsize, resample=Image.NEAREST) for m in GT['ord_masks']]
		GT['img'] = Image.open(img).resize(eval_imgsize, resample=Image.ANTIALIAS)
		GT_data[os.path.basename(img)] = GT

	for name, path in methods.items():
		if predictions.get(name, None) is None:
			results = Parallel(n_jobs=15)(delayed(process_image)(fname, name, path, eval_imgsize) for fname in tqdm(GT_data.keys()))
			# results = [process_image(fname, name, path, eval_imgsize) for fname in tqdm(GT_data.keys())]
			pred_data = {fname: pred for fname, pred in zip(GT_data.keys(), results)}
			predictions[name] = pred_data
		else:
			pred_data = predictions[name]

		os.makedirs(f"{args.out_dir}/{name}/images", exist_ok=True)
		meta_data = {'fname': [], 'pred_tax': [], 'gt_tax': []}
		for key in GT_data.keys():
			GT = GT_data[key]
			npimg = np.array(GT['img'])
			
			i = 0
			for mask, tax in zip(GT['ord_masks'], GT['ord_tax']):
				msk = np.array(mask)
				heatmap_img = cv2.applyColorMap(msk, cv2.COLORMAP_JET)
				fin = cv2.addWeighted(heatmap_img, 0.5, npimg, 0.5, 0)
				fin = plot_boundary(msk, npimg)
				cv2.imwrite(f'{args.out_dir}/{name}/images/{key[:-4]}_{i}.jpg', fin)

				i+=1
			
			pred_array = np.transpose(pred_data[key], (1, 2, 0))
			pred = pred_array.reshape(-1, pred_array.shape[-1])

			i = 0
			for mask, tax in zip(GT['ord_masks'], GT['ord_tax']):
				maskarray = np.array(mask).reshape(-1)
				vlocs = maskarray == 255
				pred_crop = pred[vlocs]
				predictions_crop = np.mean(pred_crop, axis=0)
				meta_data['fname'].append(f'{key[:-4]}_{i}.jpg')
				meta_data['pred_tax'].append(predictions_crop)
				meta_data['gt_tax'].append(tax)
				
				i += 1
		
		with open(f"{args.out_dir}/{name}/{name}_{args.split}.pkl", 'wb') as f:
			pickle.dump(meta_data, f)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='training hyper-parameters')
	parser.add_argument('--imdir', dest='imdir', type=str,
						default="./dataset")
	parser.add_argument('--split', dest='split', type=str,
						default="val")
	parser.add_argument('--config', dest='config', type=str,
						default="configs/defaults_vis.yaml")
	parser.add_argument('--out_dir', dest='out_dir', type=str,
						default="./results_html")
	args = parser.parse_args()
	main(args)
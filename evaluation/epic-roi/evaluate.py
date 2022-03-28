from __future__ import annotations
import sys, argparse, os, json
from utils import load_config, fileparts, set_numpythreads
set_numpythreads()
import numpy as np
from PIL import Image, ImageFilter
from sklearn import metrics
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pandas as pd
from collections import defaultdict


def process_image(fname, model_config, factor, sz):
	path = model_config['path']
	blurring = model_config['blurring']
	bradius = model_config.get('bradius', 0) // factor

	pred = Image.open(f"{path}/final_{fname}").resize(sz, Image.ANTIALIAS)
	if blurring:
		# Blurring the predictions
		if bradius > 0:
			pred = pred.filter(ImageFilter.GaussianBlur(radius=bradius))
	pred = np.array(pred).astype(np.float32)
	return (pred, fname)

def combine_predictions(fname, model_config1, model_config2, factor, sz):
	path1 = model_config1['path'] # MaskRCNN Predictions
	path2 = model_config2['path'] # Predictions from the other model
	
	pred_img1 = f"{path1}/final_{fname}"
	pred_img2 = f"{path2}/final_{fname}"


	pred1 = Image.open(pred_img1)
	if model_config1['blurring']:
		bradius = model_config1.get('bradius', 0)
		if bradius > 0 :
			pred1 = pred1.filter(ImageFilter.GaussianBlur(radius=bradius))
	
	pred1 = np.array(pred1).astype(np.float32)
	
	pred2 = Image.open(pred_img2)
	if model_config2['blurring']:
		bradius = model_config2.get('bradius', 0)
		if bradius > 0 :
			pred2 = pred2.filter(ImageFilter.GaussianBlur(radius=bradius))
	
	pred2 = np.array(pred2).astype(np.float32)
	
	w1 = model_config2['MaskRCNN_weight'] # Multiplier for MaskRCNN predictions
	w2 = 1 # Multiplier for the other model's predictions

	pred = (pred1*w1 + pred2*w2 )/ (w1 + w2)
	pred = np.clip(pred / 255., 0.0, 1.0)
	pred = np.array(Image.fromarray((pred * 255).astype(np.uint8)).resize(sz, Image.ANTIALIAS)).astype(np.float32)

	return (pred, fname)

def computeAP(pred_data, GT_data):
	GT_list = []
	pred_list = []
	for key in GT_data.keys():
		GT = GT_data[key].reshape(-1)
		msk = (GT == 0) | (GT == 255) # Ignore Uncertain and Human annotated regions

		GT = GT[msk]
		GT[GT<255] = 0
		GT[GT==255] = 1

		pred = pred_data[key].reshape(-1)
		pred = pred[msk] / 255.

		GT_list.append(GT)
		pred_list.append(pred)

	GTs = np.concatenate(GT_list, axis=0)
	preds = np.concatenate(pred_list, axis=0)

	# AP = metrics.average_precision_score(GTs, preds) # Uncomment if scikit-learn implementation of AP is required

	precision, recall, _ = metrics.precision_recall_curve(GTs, preds)

	idx = np.argsort(recall)
	sprecision = precision[idx]
	srecall = recall[idx]
	AP_trapz = np.trapz(sprecision, srecall) # Custom Implementation of AP

	return AP_trapz

def evaluate(config, args, frames, metadata):
	# Get all methods
	methods = config["methods"]

	# Computing the image size after downsampling
	factor = config["factor"]
	eval_imgsize = (1920 // factor, 1080 // factor)
	
	# Creating dictionary to store the metrics for each method
	result_dict = defaultdict(list)
	predictions = {}
	dilation_list = config["dilation_list"]

	for dilation_rate in dilation_list:
		print(f"Evaluating at dilation rate {dilation_rate}")

		# Read GT Images
		GT_dir = f"{args.GT_homedir}/GT{dilation_rate}_cat{args.cat}"
		GT_images = [f"{GT_dir}/{i}" for i in frames]

		print(f"Using {len(GT_images)} images for testing")
		GT_data = {}

		for img in GT_images:
			GT = np.array(Image.open(img).resize(eval_imgsize, Image.NEAREST)).astype(np.float32)
			GT_data[os.path.basename(img)] = GT

		# Read and process predictions
		for name, model_config in tqdm(methods.items()):
			# Process predictions and cache them
			if predictions.get(name, None) is None:
				results = Parallel(n_jobs=15)(delayed(process_image)(
									fname,
									model_config,
									factor, eval_imgsize
									) \
									for fname, _ in [(i, j) for i, j in GT_data.items()])
				pred_data = {fname: pred for pred, fname in results}
				predictions[name] = pred_data
			else:
				# If already processed, obtain from the cache
				pred_data = predictions[name]
			
			AP = computeAP(pred_data, GT_data)
			result_dict[name].append(round(AP, 3))

			if model_config.get("eval_with_MaskRCNN", False):
				comb_name = "MaskRCNN+" + name
				if predictions.get(comb_name, None) is None:
					results = Parallel(n_jobs=15)(delayed(combine_predictions)(
										fname,
										methods['MaskRCNN'],
										model_config,
										factor, eval_imgsize
										) \
										for fname, _ in [(i, j) for i, j in GT_data.items()])
					pred_data = {fname: pred for pred, fname in results}
					predictions[comb_name] = pred_data
				else:
					# If already processed, obtain from the cache
					pred_data = predictions[comb_name]
				
				AP = computeAP(pred_data, GT_data)
				result_dict[comb_name].append(round(AP, 3))
		print("-"*65)

	result_dict['Dilation Rate'] = dilation_list
	df = pd.DataFrame.from_dict(result_dict,orient='index')
	df.columns = df.loc['Dilation Rate']
	df = df.drop(index='Dilation Rate')
	if args.verbose:
		# Print the results
		print(f"For category {metadata['categories'][args.cat]}")
		print(df.to_string())

	df.to_csv(f"{args.out_dir}/benchmark_{args.cat}_{args.split}.csv")

def main(args):
	h, b, bwoext, ext = fileparts(args.config)
	
	# Change output directory if loading any other config file
	if b != "defaults.yaml":
		args.out_dir = os.path.join(args.out_dir, bwoext)
	
	os.makedirs(args.out_dir, exist_ok=True)

	metadata = json.load(open(args.annotated_frames, "r"))
	
	if args.split == "val":
		frames = metadata["val_images"]
	elif args.split == "test":
		frames = metadata["test_images"]
	else:
		print("Error: Use a valid split")
		sys.exit()
	
	# Loading config with paths to predictions
	config = load_config(args.config)

	# Run evaluation
	if args.cat == "all":
		# Run for all categories (as shown in the manuscript)
		categories = ["1234", "1", "2", "3", "4"]
		for cat in categories:
			args.cat = cat
			evaluate(config, args, frames, metadata)
	else:
		evaluate(config, args, frames, metadata)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='training hyper-parameters')
	parser.add_argument('--out_dir', dest='out_dir', type=str,
						default="./results/")
	parser.add_argument('--GT_homedir', dest='GT_homedir', type=str,
						default="./data/roi-data/")
	parser.add_argument('--annotated_frames', dest='annotated_frames', type=str,
						default="./data/annotated_frames.json")
	parser.add_argument('--config', dest='config', type=str,
						default="./configs/defaults.yaml")
	parser.add_argument('--split', dest='split', type=str,
						default="val")
	parser.add_argument('--cat', dest='cat', type=str,
						default="123")
	parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
	args = parser.parse_args()
	if args.split not in ["val", "test"]:
		print("Please use a valid split (val/test)")
		sys.exit()
	main(args)
import sys, os, json
from utils import load_config, fileparts, read_lzma, set_numpythreads
set_numpythreads()
from collections import defaultdict
import torch, argparse
import numpy as np
from PIL import Image
from sklearn import metrics
from torch.nn import functional as F
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
# Limit PyTorch number of threads
torch.set_num_threads(4)

def process_image(fname, name, path, n_classes, sz=None):
	if "random" in name:
		pred_GTactions = np.random.uniform(0, 1.0, (n_classes, sz[1], sz[0]))
	else:
		pred_img = f"{path}/final_{fname[:-3]}npy"
		if sz is not None:
			pred = F.interpolate(torch.from_numpy(np.load(pred_img)).permute(2, 0, 1).unsqueeze(0), sz[::-1])[0]
			pred_GTactions = torch.zeros(n_classes, sz[1], sz[0])
			for i in range(n_classes):
				pred_GTactions[i] += pred[i]
			pred_GTactions = pred_GTactions.numpy()
		else:
			pred_GTactions = np.load(pred_img)
	
	return np.round(pred_GTactions * 255, 1)

def evaluate(config, class_info, frames):
	# Reading the list of methods and eval image resolution
	methods = config['methods']
	eval_imgsize = config.get('imsize', None) # (256, 128)
	
	# Obtaining the grasp list
	GT_GRASP_LIST = class_info['easy_classes']['classes']
	num_classes = len(GT_GRASP_LIST)

	# Initializing the dictionary to store results
	result_dict = defaultdict(defaultdict)
	predictions = {}

	# Get the list of images to be evaluated
	images = [f"{args.imdir}/{args.split}/{i}" for i in frames]
	
	# Read the ground truth
	GT_data = {}
	for img in images:
		GT = read_lzma(img[:-4] + "_proc.xz")
		if eval_imgsize is not None:
			GT['ord_masks'] = [Image.fromarray(m).resize(eval_imgsize, resample=Image.NEAREST) for m in GT['ord_masks']]
		GT_data[os.path.basename(img)] = GT

	for name, path in methods.items():
		if predictions.get(name, None) is None:
			# start = time.time()
			results = Parallel(n_jobs=8)(delayed(process_image)(fname, name, path, num_classes, eval_imgsize) \
													for fname in tqdm(GT_data.keys()))
			pred_data = {fname: pred for fname, pred in zip(GT_data.keys(), results)}
			predictions[name] = pred_data
		else:
			pred_data = predictions[name]
		
		GT_list = []
		pred_list = []
		for key in GT_data.keys():
			GT = GT_data[key]
			pred_array = np.transpose(pred_data[key], (1, 2, 0)) # H X W X N
			pred = pred_array.reshape(-1, pred_array.shape[-1]) # HW X N
			for mask, tax in zip(GT['ord_masks'], GT['ord_tax']):
				# Get the object Mask
				maskarray = np.array(mask).reshape(-1)
				vlocs = maskarray == 255
				pred_crop = pred[vlocs]
				
				# Obtain the predictions at the object mask
				predictions_crop = np.mean(pred_crop, axis=0)
				pred_list.append(predictions_crop)

				# Get the grasps applicable to the object
				GT_list.append([i - 1 for i in tax])

		# Convert the applicable grasps to binary vectors
		mlb = MultiLabelBinarizer(classes=np.arange(num_classes))
		labels = mlb.fit_transform(GT_list)
		
		# Obtaining the score
		scores = np.stack(pred_list, axis=0)
		if "random" in name.lower():
			scores = np.random.uniform(0, 1, labels.shape)
		APs = {}
		for i in range(labels.shape[1]):
			if np.mean(labels[:, i]) == 0:
				AP = 0
			else:
				AP = metrics.average_precision_score(labels[:, i], scores[:, i])
				if "random" in name.lower():	
					# Get the random baseline
					AP = np.mean(labels[:, i])

				APs[i] = AP
				result_dict[name][f"AP_{i}"] = AP
		
		mAP = np.mean([j for i, j in APs.items() if j != 0])
		result_dict[name]['mAP'] = mAP
	return result_dict

def main(args):
	h, b, bwoext, ext = fileparts(args.config)
	# Change output directory if loading any other config file
	if b != "defaults.yaml":
		args.out_dir = os.path.join(args.out_dir, bwoext)
	os.makedirs(args.out_dir, exist_ok=True)
	
	# Load config and class information file
	config = load_config(args.config)
	class_info = load_config(args.class_info)

	# Load Metadata
	metadata = json.load(open(args.annotated_frames, "r"))
	
	if args.split == "val":
		frames = metadata["val_images"]
	elif args.split == "test":
		frames = metadata["test_images"]
	else:
		print("Error: Use a valid split")
		sys.exit()

	# evaluate the predictions
	result_dict = evaluate(config, class_info, frames)

	if args.verbose:
		# Print the metrics
		for name in result_dict.keys():
			print(f"{name}: mAP: {result_dict[name]['mAP']:.3f}")
	
	frames = []
	for m in result_dict.keys():
		frames.append(pd.DataFrame.from_dict(result_dict[m], orient='index', columns=[m]))

	results_frame = pd.concat(frames, axis=1)
	# Dumping the results to csvs
	results_frame.to_csv(f"{args.out_dir}/benchmark_{args.split}.csv", float_format='%.3f')
	results_frame.T.to_csv(f"{args.out_dir}/benchmarktp_{args.split}.csv", float_format='%.3f')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='training hyper-parameters')
	parser.add_argument('--imdir', dest='imdir', type=str,
						default="./dataset/")
	parser.add_argument('--split', dest='split', type=str,
						default="val")
	parser.add_argument('--annotated_frames', dest='annotated_frames', type=str,
						default="./dataset/annotated_frames.json")
	parser.add_argument('--config', dest='config', type=str,
						default="./configs/defaults.yaml")
	parser.add_argument('--class_info', dest='class_info', type=str,
						default="./configs/class_info.yaml")
	parser.add_argument('--out_dir', dest='out_dir', type=str,
						default="./results/")
	parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
	args = parser.parse_args()
	if args.split not in ["val", "test"]:
		print("Please use a valid split (val/test)")
		sys.exit()
	main(args)
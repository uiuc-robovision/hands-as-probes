import sys
sys.path.append('../')
from helper import set_numpythreads, str2bool, get_git_root
set_numpythreads()
import os, json
from PIL import Image, ImageEnhance
import torch
import numpy as np
from torchvision import transforms
import cv2
from tqdm import tqdm
import argparse

# Limit CPU usage by pytorch
torch.set_num_threads(4)

# global transformation
to_tensor = transforms.ToTensor()
to_PIL = transforms.ToPILImage()
color_map = {'red':[1,0,0], 'green':[0,1,0], 'blue':[0,0,1],
			'cyan':[0,1,1], 'magenta':[1,0,1], 'yellow':[1,1,0]}

def add_blur(tensor, sz, Z): # (3, 224, 224)
	tensor = tensor.permute(1,2,0).numpy()
	k_size = int(np.sqrt(sz**2) / Z)
	if k_size % 2 == 0:
		k_size += 1
	tensor = cv2.GaussianBlur(tensor, (k_size, k_size), 0)
	tensor = torch.from_numpy(tensor).permute(2,0,1)
	return tensor

def post_process(hmaps, blur=True):
	hmaps = torch.stack([hmap/(hmap.max() + 1e-12) for hmap in hmaps], 0)
	hmaps = hmaps.numpy()

	processed = []        
	for c in range(hmaps.shape[0]):
		hmap = hmaps[c]
		if blur:
			hmap = cv2.GaussianBlur(hmap, (3, 3), 0)
		processed.append(hmap)
	processed = np.array(processed)
	processed = torch.from_numpy(processed).float()

	return processed

def generate_color_map(hmaps, colors, iblur=True):
	colors = [color_map[c] for c in colors]
	colors = 1 - torch.FloatTensor(colors).unsqueeze(2).unsqueeze(2) # invert colors

	vals, idx = torch.sort(hmaps, 0, descending=True)
	cmap = torch.zeros(hmaps.shape)
	for c in range(hmaps.shape[0]):
		cmap[c][idx[0]==c] = vals[0][idx[0]==c]

	cmap = cmap.unsqueeze(1).expand(cmap.shape[0], 3, cmap.shape[-2], cmap.shape[-1]) # (C, 3, 224, 224)
	cmap = [hmap*color for hmap, color in zip(cmap, colors)]
	cmap = torch.stack(cmap, 0) # (C, 3, 14, 14)

	cmap, _ = cmap.max(0)

	# blur the heatmap to make it smooth
	if iblur:
		cmap = add_blur(cmap, cmap.shape[1], 25) # blur(cmap, cmap.shape[1], 9)
	cmap = 1 - cmap # invert heatmap: white background

	# improve contrast for visibility
	cmap = transforms.ToPILImage()(cmap)
	cmap = ImageEnhance.Color(cmap).enhance(1.5)
	cmap = ImageEnhance.Contrast(cmap).enhance(1.5)
	cmap = transforms.ToTensor()(cmap)

	return cmap


def overlay_colored_heatmaps(uimg, hmaps, blur=True): # (C, 224, 224)
	colors = ['red']
	# post process heatmaps: normalize each channel, blur, threshold
	phmaps = post_process(hmaps, blur) # (C, 224, 224) 

	# generate color map from each heatmap channel
	cmap = generate_color_map(phmaps, colors, blur)

	# generate per-pixel alpha channel and overlay
	alpha = (1-cmap).mean(0)
	overlay = (1-alpha)*uimg + alpha*cmap

	return overlay

def create_vis(img_path, hm_path):
	img = Image.open(img_path)
	hm = Image.open(hm_path)

	# create tensors
	imgtensor = to_tensor(img)
	hmtensor = to_tensor(hm)

	# create overlay
	overlay = overlay_colored_heatmaps(imgtensor, hmtensor, True)

	ol = to_PIL(overlay)
	img = to_PIL(imgtensor)

	return img, ol

def main(args):
	model_name = args.model_name
	pred_dir = f"{args.pred_dir}/{model_name}"
	
	out_dir = f"{args.out_dir}/{model_name}"
	os.system(f"mkdir -p {out_dir}")

	# Load metadata containing split info and image names
	metadata = json.load(open(args.annotated_frames, "r"))
	
	if args.split == "val":
		frames = metadata["val_images"]
	elif args.split == "test":
		frames = metadata["test_images"]
	else:
		print("Error: Use a valid split")
		sys.exit()

	# Get path to images and predictions
	im_list = [f"{args.inp_dir}/{i[:-3]}jpg" for i in frames]
	hm_list = [pred_dir + "/final_" + os.path.basename(i)[:-3] + "png" for i in im_list]

	for i, j in tqdm(zip(im_list, hm_list), total=len(im_list)):
		fname = os.path.splitext(os.path.basename(i))[0]
		
		# overlay heatmaps
		img, hm = create_vis(i, j)
		
		# resize the images and overlayed heatmaps before saving output
		img.resize((456, 256), resample=Image.ANTIALIAS).save(f"{out_dir}/{fname}_imgsmall.jpg")
		hm.resize((456, 256), resample=Image.ANTIALIAS).save(f"{out_dir}/{fname}_hmsmall.png")

		# Uncomment to save original overlays
		# img.save(f"{out_dir}/{fname}_img.png")
		# hm.save(f"{out_dir}/{fname}_hm.png")




if __name__ == "__main__":
	GIT_ROOT = get_git_root(__file__)
	parser = argparse.ArgumentParser(description='Visualization parameters')
	parser.add_argument('--model_name', dest='model_name', type=str,
						default="SegNet_hands_seed0_")
	parser.add_argument('--pred_dir', dest='pred_dir', type=str,
						default=f"{GIT_ROOT}/training/inferred_IHdata/epic_benchmark")
	parser.add_argument('--annotated_frames', dest='annotated_frames', type=str,
						default=f"{GIT_ROOT}/evaluation/epic-roi/data/annotated_frames.json")
	parser.add_argument('--inp_dir', dest='inp_dir', type=str,
						default=f"{GIT_ROOT}/evaluation/epic-roi/data/roi-data/images")
	parser.add_argument('--split', dest='split', type=str,
						default="val")
	parser.add_argument('--out_dir', dest='out_dir', type=str,
						default="./inferred_vis/epic")

	args = parser.parse_args()
	main(args)

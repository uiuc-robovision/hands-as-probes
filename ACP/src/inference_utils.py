import os, sys
import numpy as np

import torch, pickle
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, ColorJitter, Normalize, RandomHorizontalFlip, RandomAffine
import torchvision.transforms.functional as TF
from box_utils import BBox, draw_bbox
from transforms import resize_image, nparray2img
import cv2

def to_int(x):
	return int(np.floor(x))

def load_meta_data(annot_dir, split):
	file = os.path.join(annot_dir, split + "_contact" + ".pkl")
	# file = os.path.join(annot_dir, split + ".pkl")
	with open(file, "rb") as f:
		meta_info = pickle.load(f)
	return meta_info

def create_bbox_fromhand(hand, img):
		if hand is None:
			return None
		return BBox(hand[0], hand[1], hand[2], hand[3], img, True)

def get_imgnet_transform():
	normalize = Normalize(mean=[0.485, 0.456, 0.406],
							  std=[0.229, 0.224, 0.225])
	transform = Compose([ToTensor(), normalize])

	return transform

def vis_prediction(probs, vmask, bbox, img, path=None):
	npimg = np.array(img)
	empty_image_seg = np.zeros((npimg.shape[0], npimg.shape[1]))
	empty_image_counts = np.zeros((npimg.shape[0], npimg.shape[1]))

	crop_seg = nparray2img(probs)
	bbox = bbox.scale(2.0)
	bbox = bbox.shift(shiftv=bbox.height/2.)
	resized_crop_seg = resize_image(crop_seg, to_int(bbox.width), to_int(bbox.height))

	npcrop_seg = np.array(resized_crop_seg) / 255.
	npvalidity_mask = np.ones((npcrop_seg.shape)).astype(np.float32)

	empty_image_seg = add_to_image(empty_image_seg, npcrop_seg * npvalidity_mask, bbox)
	empty_image_counts = add_to_image(empty_image_counts, npvalidity_mask, bbox)

	final_npimage = empty_image_seg / (empty_image_counts + 1e-5)

	heatmap_img = cv2.applyColorMap((final_npimage*255.).astype(np.uint8), cv2.COLORMAP_JET)

	dimg = draw_bbox(img, bbox, "white")
	fin = cv2.addWeighted(heatmap_img, 0.5, npimg, 0.5, 0)

	if path is not None:
		dimg.save(f"{path}_bbox.png")
		cv2.imwrite(f'{path}_hm.jpg', fin)

	return dimg, fin

def load_pickle(fp):
	return pickle.load(open(fp, "rb"))


def get_scores(embs, ccenters):
	sim = F.cosine_similarity(embs.unsqueeze(1), ccenters.unsqueeze(0), dim=-1)
	return sim


def add_to_image(img, crop, bbox):
	pr_l = max(0, to_int(bbox.left))
	pr_t = max(0, to_int(bbox.top))
	pr_r = min(to_int(bbox.max_width), to_int(bbox.right))
	pr_b = min(to_int(bbox.max_height), to_int(bbox.bottom))
	
	bw = to_int(bbox.right) - to_int(bbox.left)
	bh = to_int(bbox.bottom) - to_int(bbox.top)

	tpr_l = pr_l - to_int(bbox.left)
	tpr_t = pr_t - to_int(bbox.top)
	tpr_r = bw + pr_r - to_int(bbox.right)
	tpr_b = bh + pr_b - to_int(bbox.bottom)
	
	if pr_b - pr_t > tpr_b - tpr_t:
		pr_b = pr_t + tpr_b - tpr_t
	elif pr_b - pr_t < tpr_b - tpr_t:
		tpr_b = tpr_t + pr_b - pr_t


	if pr_r - pr_l > tpr_r - tpr_l:
		pr_r = pr_l + tpr_r - tpr_l
	elif pr_r - pr_l < tpr_r - tpr_l:
		tpr_r = tpr_l + pr_r - pr_l 

	img[pr_t:pr_b, pr_l:pr_r] += crop[tpr_t:tpr_b, tpr_l:tpr_r]
	
	return img


def fill_action_heatmap(img, scores, bb_list, num_clusters):
	npimg = np.array(img)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	empty_image_seg = np.zeros((npimg.shape[0], npimg.shape[1], num_clusters))
	empty_image_counts = np.zeros((npimg.shape[0], npimg.shape[1], num_clusters))
	# if args.vis:
	# 	for ind in range(len(npprobs)):
	# 		os.system(f"mkdir -p {out_dir}/{item_name}")
	# 		dimg, fin = vis_prediction(npprobs[ind, :, :, 0], vm_list[ind], bb_list[ind], img, f"{out_dir}/{item_name}/{str(ind).zfill(3)}")

	for ind in range(len(scores)):
		# crop_seg = probs[ind:ind+1]
		crop_seg = torch.ones(1, 1, 100, 100).to(device)
		aprob = scores[ind].to(device)
		bbox = bb_list[ind]

		bw = to_int(bbox.right) - to_int(bbox.left)
		bh = to_int(bbox.bottom) - to_int(bbox.top)
		resized_crop_seg = F.interpolate(crop_seg, (bw, bh), mode="bilinear") # resize_image(crop_seg, bw, bh)
		# npvalidity_mask = np.array(vm_list[ind]) / 255.
		npcrop_seg = resized_crop_seg[0].repeat(num_clusters, 1, 1) * aprob.view(-1, 1, 1)
		npcrop_seg = npcrop_seg.permute(1, 2, 0).cpu().numpy()
		npvalidity_mask = np.ones((npcrop_seg.shape)).astype(np.float32)

		empty_image_seg = add_to_image(empty_image_seg, npcrop_seg * npvalidity_mask, bbox)
		empty_image_counts = add_to_image(empty_image_counts, npvalidity_mask, bbox)

	# nparray2img(empty_image_counts).save(f"{out_dir}/counts_{item_name}.jpg")
	final_npimage_ = empty_image_seg / (empty_image_counts + 1e-5)

	return final_npimage_
	
def fill_action_heatmapGPU(img, scores, bb_list, num_clusters):
	npimg = np.array(img)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	empty_image_seg = torch.zeros((npimg.shape[0], npimg.shape[1], num_clusters)).to(device)
	empty_image_counts = torch.zeros((npimg.shape[0], npimg.shape[1], num_clusters)).to(device)
	# if args.vis:
	# 	for ind in range(len(npprobs)):
	# 		os.system(f"mkdir -p {out_dir}/{item_name}")
	# 		dimg, fin = vis_prediction(npprobs[ind, :, :, 0], vm_list[ind], bb_list[ind], img, f"{out_dir}/{item_name}/{str(ind).zfill(3)}")

	for ind in range(len(scores)):
		# crop_seg = probs[ind:ind+1]
		crop_seg = torch.ones(1, 1, 100, 100).to(device)
		aprob = scores[ind].to(device)
		bbox = bb_list[ind]

		bw = to_int(bbox.right) - to_int(bbox.left)
		bh = to_int(bbox.bottom) - to_int(bbox.top)
		resized_crop_seg = F.interpolate(crop_seg, (bw, bh), mode="bilinear") # resize_image(crop_seg, bw, bh)
		# npvalidity_mask = np.array(vm_list[ind]) / 255.
		npcrop_seg = resized_crop_seg[0].repeat(num_clusters, 1, 1) * aprob.view(-1, 1, 1)
		npcrop_seg = npcrop_seg.permute(1, 2, 0)
		npvalidity_mask = torch.ones((npcrop_seg.shape)).to(device)

		empty_image_seg = add_to_image(empty_image_seg, npcrop_seg * npvalidity_mask, bbox)
		empty_image_counts = add_to_image(empty_image_counts, npvalidity_mask, bbox)

	# nparray2img(empty_image_counts).save(f"{out_dir}/counts_{item_name}.jpg")
	final_npimage_ = empty_image_seg / (empty_image_counts + 1e-5)

	return final_npimage_.cpu().numpy()
	

def fill_object_heatmap(img, probs, bb_list):
	# npprobs = probs.numpy()
	npimg = np.array(img)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	empty_image_seg = np.zeros((npimg.shape[0], npimg.shape[1]))
	empty_image_counts = np.zeros((npimg.shape[0], npimg.shape[1]))

	for ind in range(len(probs)):
		# crop_seg = nparray2img(probs[ind].numpy()[0, :, :, 0])
		crop_seg = probs[ind].to(device).permute(0, 3, 1, 2)
		bbox = bb_list[ind]

		bw = to_int(bbox.right) - to_int(bbox.left)
		bh = to_int(bbox.bottom) - to_int(bbox.top)
		# resized_crop_seg = resize_image(crop_seg, bw, bh)
		# npvalidity_mask = np.array(vm_list[ind]) / 255.
		# npcrop_seg = np.array(resized_crop_seg) / 255.
		npcrop_seg = F.interpolate(crop_seg, (bw, bh), mode="bilinear").squeeze(0).squeeze(0).cpu().numpy() # resize_image(crop_seg, bw, bh)

		empty_image_seg = add_to_image(empty_image_seg, npcrop_seg, bbox)
		empty_image_counts = add_to_image(empty_image_counts, np.ones((npcrop_seg.shape)).astype(np.float32), bbox)

	final_npimage_ = empty_image_seg / (empty_image_counts + 1e-5)

	return final_npimage_

def fill_object_heatmapGPU(img, probs, bb_list):
	# npprobs = probs.numpy()
	npimg = np.array(img)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	empty_image_seg = torch.zeros((npimg.shape[0], npimg.shape[1])).to(device)
	empty_image_counts = torch.zeros((npimg.shape[0], npimg.shape[1])).to(device)

	for ind in range(len(probs)):
		# crop_seg = nparray2img(probs[ind].numpy()[0, :, :, 0])
		crop_seg = probs[ind].to(device).permute(0, 3, 1, 2)
		bbox = bb_list[ind]

		bw = to_int(bbox.right) - to_int(bbox.left)
		bh = to_int(bbox.bottom) - to_int(bbox.top)
		# resized_crop_seg = resize_image(crop_seg, bw, bh)
		# npvalidity_mask = np.array(vm_list[ind]) / 255.
		# npcrop_seg = np.array(resized_crop_seg) / 255.
		npcrop_seg = F.interpolate(crop_seg, (bw, bh), mode="bilinear").squeeze(0).squeeze(0) # resize_image(crop_seg, bw, bh)

		empty_image_seg = add_to_image(empty_image_seg, npcrop_seg, bbox)
		empty_image_counts = add_to_image(empty_image_counts, torch.ones((npcrop_seg.shape)).to(device), bbox)

	final_npimage_ = empty_image_seg / (empty_image_counts + 1e-5)

	return final_npimage_.cpu().numpy()
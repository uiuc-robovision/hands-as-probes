from helper import set_numpythreads, str2bool, get_git_root
set_numpythreads()
import os, json, sys
import cv2
import torch
from utils import load_config
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import glob

from model import SegmentationNetDeeper, SegmentationNetDeeperBig, SegmentationNetDeeperTwohead, SegmentationNetDeeperTwoheadBig
from transforms import crop_image, resize_image, nparray2img
from transforms import generate_bboxes_around_forimg, create_validity_mask
from inference_utils import fill_object_heatmapGPU, fill_action_heatmapGPU, get_imgnet_transform, vis_prediction

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.set_num_threads(4)

def get_data_for_image_path(item, GT_item, imsize, transform, mask_location, wsize=None, hand_cond=True, fix_target=False):
	img = Image.open(item)
	if hand_cond:
		GT = Image.open(GT_item)

	bbox_list = []
	inp_tenor_list = []
	vmasks_list = []

	# Create validity mask based on mask location and masking argument
	validity_mask = create_validity_mask(imsize, masking=args.mask_input, mask_location=mask_location)

	validity_mask_tensor = torch.from_numpy(validity_mask).type(torch.float32).unsqueeze_(0)
	validity_img = nparray2img(validity_mask_tensor.numpy()[0, :, :])
	exp_val_mask = np.expand_dims(validity_mask.astype(np.uint8), -1)
	
	# Add margin if the patch location should be at the center (empirically worked better)
	margin = 0.5 if mask_location == "center" else 0.0

	if wsize is None:
		# Multiscale inference when wsize is set to None
		boxes = [] \
					+ generate_bboxes_around_forimg(img.copy(), 30, 30, nh=80, nv=50, margin=margin) \
					+ generate_bboxes_around_forimg(img.copy(), 50, 50, nh=80, nv=50, margin=margin) \
					+ generate_bboxes_around_forimg(img.copy(), 80, 80, nh=80, nv=50, margin=margin)
	else:
		boxes = generate_bboxes_around_forimg(img.copy(), wsize, wsize, nh=80, nv=50, margin=margin)
	
	for i in range(len(boxes)):
		outs = boxes[i]
		bbox, bbaround = outs

		if mask_location == "center":
			# Originally patch is at the bottom center, shift the bigger patch down to have masking at the center
			bbaround = bbaround.shift(shiftv=-bbox.height/2.)

		if hand_cond:
			GT_crop = np.array(crop_image(GT, bbaround)).astype(np.float32)
			# Skip the patch if it overlaps with hand regions
			if np.sum(GT_crop == 128) > 0:
				continue
		
		inp_img = resize_image(crop_image(img, bbaround), imsize, imsize)
		inp_img = Image.fromarray(np.array(inp_img) * exp_val_mask) # Hide Hand region

		inp_img_tensor = transform(inp_img)
		inp_tenor_list.append(inp_img_tensor)
		
		if not fix_target:
			# Used when returning box where the ROI predictions are pasted
			if args.sym_encdec:
				if not args.loss_masking:
					bbox_list.append(bbaround)
				else:
					bbox_list.append(bbox)
			else:
				bbox_list.append(bbox)
		else:
			# Used when returning box where the hand predictions are pasted
			bbox_list.append(bbox)

		vmasks_list.append(validity_img)
	
	return img, bbox_list, vmasks_list, torch.stack(inp_tenor_list, dim=0)

@torch.no_grad() # Make the function no grad
def infer(out_dir, model_ROI, model_YCB, n_classes, mask_location, args):
	# Set batchsize and device
	bs = 512
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model_ROI = model_ROI.to(device)
	model_YCB = model_YCB.to(device)

	# define transforms
	transform = get_imgnet_transform()

	# Get images
	types = ('*.png', '*.jpg') # the tuple of file types
	images = []
	for files in types:
		images.extend(sorted(glob.glob(f"{args.inp_dir}/{files}")))

	for item in tqdm(images):
		img, bb_list, _, inp_ims = get_data_for_image_path(item, None, args.imsize, transform, mask_location, wsize=args.wsize, hand_cond=False)
		item_name = os.path.splitext(os.path.basename(item))[0]

		out_list = []
		with torch.no_grad():
			for i in range(0, int(np.ceil(len(inp_ims) / bs))):
				out = model_ROI.infer_cshifted(inp_ims[i*bs:(i+1)*bs].to(device))
				p = out['pred'].permute(0, 2, 3, 1).cpu()
				if args.sym_encdec:
					if args.loss_masking:
						imsize = args.imsize
						if mask_location == "bc":
							p = p[:, imsize * 2 // 4: imsize * 4 // 4 , imsize * 1 // 4: imsize * 3 // 4, :]
						elif mask_location == "center":
							p = p[:, imsize * 1 // 4: imsize * 3 // 4 , imsize * 1 // 4: imsize * 3 // 4, :]
						else:
							print("Invalid Masking location")
							exit()
				out_list += list(torch.split(p, 1))

			probs = out_list
		
		object_image = fill_object_heatmapGPU(img, probs, bb_list)
		
		# Convert Image to Numpy Array
		npimg = np.array(img)

		# Save ROI Output
		heatmap_img = cv2.applyColorMap((object_image*255.).astype(np.uint8), cv2.COLORMAP_JET)
		fin = cv2.addWeighted(heatmap_img, 0.5, npimg, 0.5, 0)
		cv2.imwrite(f'{out_dir}/color_img_{item_name}.jpg', fin)

		img, bb_list, _, inp_ims = get_data_for_image_path(item, None, args.imsize, transform, mask_location, wsize=args.wsize, hand_cond=False, fix_target=True)
		
		# Convert Image to Numpy Array
		npimg = np.array(img)

		hand_pred = []
		with torch.no_grad():
			for i in range(0, int(np.ceil(len(inp_ims) / bs))):
				out = model_YCB.infer_cshiftedgrasp(inp_ims[i*bs:(i+1)*bs].to(device))
				hand_pred.append(out['hand_emb'].reshape(len(out['hand_emb']), -1).cpu())

			hand_embs = torch.cat(hand_pred, dim=0)
			scores = torch.sigmoid(hand_embs) # Convert to probabilities

		# Aggregate the predictions
		action_image = fill_action_heatmapGPU(img, scores, bb_list, n_classes)

		final_npimage_ = np.expand_dims(object_image, axis=-1) * action_image

		np.save(f"{out_dir}/final_{item_name}.npy", final_npimage_)
		img.save(f"{out_dir}/input_{item_name}.jpg")

def load_ckpt(checkpoint, args, return_masklc=False):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	mask_location = checkpoint['config']['training']['mask_location']
	if args.mask_input != checkpoint['config']['training']['masking']:
		# Change mask input argument if the model was trained with a different config
		args.mask_input = checkpoint['config']['training']['masking']
		print(f"Changing args.mask_input to {args.mask_input}")
	print(f"Masking set to {args.mask_input}, Mask location {mask_location}")

	if args.sym_encdec != checkpoint['config']['model'].get('is_sym', False):
		# Change mask input argument if the model was trained with a different config
		args.sym_encdec = checkpoint['config']['model'].get('is_sym', False)
		args.loss_masking = checkpoint['config']['training']['loss_masking']
		print(f"Changing args.sym_encdec to {args.sym_encdec}")
	print(f"Architecture symmetric is {args.sym_encdec}, loss_masking set to {args.loss_masking}")

	if args.two_heads:
		grasp_info = load_config(args.grasp_info)
		n_classes = len(grasp_info["easy_classes"]["classes"])
		if args.sym_encdec:
			model = SegmentationNetDeeperTwoheadBig(checkpoint['config']['model'], emb_size=n_classes).to(device)
		else:
			model = SegmentationNetDeeperTwohead(checkpoint['config']['model'], emb_size=n_classes).to(device)
	else:
		if args.sym_encdec:
			model = SegmentationNetDeeperBig(checkpoint['config']['model']).to(device)
		else:
			model = SegmentationNetDeeper(checkpoint['config']['model']).to(device)

	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval() # Set in eval mode
	
	if return_masklc:
		return model, mask_location

	return model

def main(args):
	model_name = args.model_name
	prefix = ""
	prefix = prefix + f"_ROI{args.ckpt_ROI}" if args.ckpt_ROI is not None else prefix
	prefix = prefix + f"_YCB{args.ckpt_YCB}" if args.ckpt_YCB is not None else prefix
	prefix = prefix + f"_{args.wsize}" if args.wsize is not None else prefix
	# prefix = prefix + f"_nomask" if not args.mask_input else prefix 
	if args.ckpt_ROI is None:
		model_path_ROI = f"{args.model_dir}/{model_name}_checkpoint.pth"
	else:
		model_path_ROI = f"{args.model_dir}/{model_name}_checkpoint_{args.ckpt_ROI}.pth"
	checkpoint_ROI = torch.load(model_path_ROI)
	
	if args.ckpt_YCB is None:
		model_path_YCB = f"{args.model_dir}/{model_name}_checkpoint.pth"
	else:
		model_path_YCB = f"{args.model_dir}/{model_name}_checkpoint_{args.ckpt_YCB}.pth"
	checkpoint_YCB = torch.load(model_path_YCB)

	out_dir = f"{args.out_dir}_benchmark/{model_name}{prefix}"

	os.system(f"mkdir -p {out_dir}")

	model_ROI, mask_location = load_ckpt(checkpoint_ROI, args, return_masklc=True)
	model_YCB = load_ckpt(checkpoint_YCB, args)

	grasp_info = load_config(args.grasp_info)
	n_classes = len(grasp_info["easy_classes"]["classes"])

	infer(out_dir, model_ROI, model_YCB, n_classes, mask_location, args)
		
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Inference parameters')
	parser.add_argument('--imsize', dest='imsize', type=int,
						default=128)
	parser.add_argument('--wsize', dest='wsize', type=int,
						default=None)
	parser.add_argument('--model_dir', dest='model_dir', type=str,
						default="./models/EPIC-KITCHENS/")
	parser.add_argument('--model_name', dest='model_name', type=str,
						default="SegNet_seed0")
	parser.add_argument('--grasp_info', dest='grasp_info', type=str,
						default="./src/configs/grasp_info.yaml")
	parser.add_argument('--inp_dir', dest='inp_dir', type=str,
						default="./demo-gun71/data")
	parser.add_argument('--out_dir', dest='out_dir', type=str,
						default="./inferred_IHdata/ACP_demo")
	parser.add_argument('--ckpt_ROI', dest='ckpt_ROI', type=int,
						default=None)
	parser.add_argument('--ckpt_YCB', dest='ckpt_YCB', type=int,
						default=None)
	parser.add_argument('--mask_input', dest='mask_input', type=str2bool, default=True)
	parser.add_argument('--sym_encdec', dest='sym_encdec', type=str2bool, default=False)
	parser.add_argument('--two_heads', dest='two_heads', type=str2bool, default=False)
	parser.add_argument('--loss_masking', dest='loss_masking', type=str2bool, default=False)

	args = parser.parse_args()
	main(args)

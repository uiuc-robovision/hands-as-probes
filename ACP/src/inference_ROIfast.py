import os, json, sys
from helper import set_numpythreads, str2bool, get_git_root
set_numpythreads()
import numpy as np
import argparse
from tqdm import tqdm
import PIL
from PIL import Image
import glob

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, ColorJitter, Normalize, RandomHorizontalFlip, RandomAffine
import torchvision.transforms.functional as TF
from model import SegmentationNetDeeper, SegmentationNetDeeperBig, SegmentationNetDeeperTwohead, SegmentationNetDeeperTwoheadBig
from utils import load_config
from transforms import nparray2img, create_validity_mask, pad_image, unpad_image
import cv2
from inference_utils import get_imgnet_transform

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.set_num_threads(4)

buffer = {}


def paste_to_canvas(predictions, fold_operation):
	preds = predictions.permute(1,2,3,0).reshape(-1,predictions.shape[0]).unsqueeze(0)
	output = fold_operation(preds)

	return output

def get_patches(item, GT_item, imsize, transform, mask_location, wsize=80, hand_cond=True, device=None):
	orig_img = img = Image.open(item)
	if hand_cond:
		GT = Image.open(GT_item)
	
	orig_h, orig_w = img.height, img.width

	# Create validity mask
	validity_mask = create_validity_mask(imsize, masking=args.mask_input, mask_location=mask_location)
	validity_mask_tensor = torch.from_numpy(validity_mask).type(torch.float32).unsqueeze_(0)
	
	
	rescale_factor = imsize * 1. / (2. * wsize)
	width, height = int(img.width*rescale_factor), int(img.height*rescale_factor)

	img = img.resize((width, height),  PIL.Image.LANCZOS)
	if hand_cond:
		GT = GT.resize((width, height),  PIL.Image.NEAREST)
		
	# Add relevant padding (left, right, top, bottom)
	padding = (0, 0, 0, 0)
	if mask_location == "bc":
		padding = (imsize//4, imsize//4, imsize//2, 0)
	img = pad_image(img, pad=padding, fill=(0,0,0))
	if hand_cond:
		GT = pad_image(GT, pad=padding, fill=0)

	width, height = img.width, img.height

	npatchesw = 80
	w = imsize
	stridew = int((width - w - 2) / (npatchesw - 1))

	npatchesh = 50
	w = imsize
	strideh = int((height - w - 2) / (npatchesh - 1))


	# Unfold Operation
	unfold = torch.nn.Unfold((imsize, imsize), dilation=1, padding=0, stride=(strideh, stridew)).to(device)
	
	tf_img = transform(img).unsqueeze(0).to(device)
	patches = unfold(tf_img)

	
	zero_val = transform(Image.fromarray(np.zeros((1, 1, 3)).astype(np.uint8))).to(device)
	patches = patches.reshape(3, imsize, imsize, -1).permute(3, 0, 1, 2)
	
	if hand_cond:
		GT_tf_img = torch.from_numpy(np.array(GT)).to(device).unsqueeze(0).unsqueeze(0)
		Hand_mask = (GT_tf_img == 128) * 1.0
		valid_patches = unfold(Hand_mask).reshape(1, imsize, imsize, -1).permute(3, 0, 1, 2)
		valid_indices = valid_patches.mean(-1).mean(-1).mean(-1) > 0
	else:
		valid_indices = torch.zeros((patches.shape[0])).to(device) > 0

	# Mask the bottom center of the sampled patches
	if args.mask_input:
		# Fix this, when the mask location changes
		if mask_location == "bc":
			patches[:, :, imsize * 2 // 4: imsize * 4 // 4 , imsize // 4: imsize * 3 // 4] = zero_val[None, ...]
		elif mask_location == "center":
			patches[:, :, imsize * 1 // 4: imsize * 3 // 4 , imsize // 4: imsize * 3 // 4] = zero_val[None, ...]
		else:
			print("Invalid Masking Location")
			exit()
	
	# Fold function
	fold = torch.nn.Fold((height, width), (imsize, imsize), dilation=1, padding=0, stride=(strideh, stridew)).to(device)
	
	return patches, validity_mask_tensor, fold, (orig_w, orig_h), orig_img, valid_indices, padding


def infer(out_dir, model, mask_location, frames, args):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	im_dir = args.inp_dir
	bs = 512

	# define transforms
	transform = get_imgnet_transform()

	# Get images
	images = [f"{args.inp_dir}/{i[:-3]}jpg" for i in frames]

	with torch.no_grad():
		for item in tqdm(images):
			
			GT_item = os.path.join(args.GT_dir, os.path.basename(item)[:-3] + "png") if args.hand_cond else None

			pred_list = []
			count_list = []

			# multi scale aggregation over patches of size 80, 50, and 30
			for wsize in [80, 50, 30]:
				patches, _, fold_op, orig_size, img, ivindices, padding = get_patches(item, GT_item, args.imsize, transform, mask_location, wsize=wsize, hand_cond=args.hand_cond, device=device)
				item_name = os.path.splitext(os.path.basename(item))[0]

				out_list = []
				for i in range(0, int(np.ceil(len(patches) / bs))):
					out = model.infer_cshifted(patches[i*bs:(i+1)*bs].to(device))
					p = out['pred']
					out_list.append(p)

				# grid_img = torchvision.utils.save_image(probs, f"{out_dir}/probs_{item_name}.png")
				probs = torch.cat(out_list, dim=0)

				npimg = np.array(img)
				imsize = args.imsize
				counts = torch.ones(probs.shape).to(device)
				
				full_predictions = torch.zeros((patches.shape[0], 1, imsize, imsize)).to(device)
				full_counts = torch.zeros((patches.shape[0], 1, imsize, imsize)).to(device)
				if not args.sym_encdec:
					if mask_location == "bc":
						full_predictions[:, :, imsize * 2 // 4: imsize * 4 // 4 , imsize * 1 // 4: imsize // 4 * 3] = probs
						full_counts[:, :, imsize * 2 // 4: imsize * 4 // 4 , imsize * 1 // 4: imsize // 4 * 3] = counts
					elif mask_location == "center":
						full_predictions[:, :, imsize * 1 // 4: imsize * 3 // 4 , imsize * 1 // 4: imsize * 3 // 4] = probs
						full_counts[:, :, imsize * 1 // 4: imsize * 3 // 4 , imsize * 1 // 4: imsize * 3 // 4] = counts
					else:
						print("Invalid Masking Location")
						exit()
				else:
					if args.loss_masking:
						if mask_location == "bc":
							full_predictions[:, :, imsize * 2 // 4: imsize * 4 // 4 , imsize * 1 // 4: imsize // 4 * 3] =\
								 probs[:, :, imsize * 2 // 4: imsize * 4 // 4 , imsize * 1 // 4: imsize // 4 * 3]
							full_counts[:, :, imsize * 2 // 4: imsize * 4 // 4 , imsize * 1 // 4: imsize // 4 * 3] =\
								 counts[:, :, imsize * 2 // 4: imsize * 4 // 4 , imsize * 1 // 4: imsize // 4 * 3]
						elif mask_location == "center":
							full_predictions[:, :, imsize * 1 // 4: imsize * 3 // 4 , imsize * 1 // 4: imsize * 3 // 4] =\
								 probs[:, :, imsize * 1 // 4: imsize * 3 // 4 , imsize * 1 // 4: imsize * 3 // 4]
							full_counts[:, :, imsize * 1 // 4: imsize * 3 // 4 , imsize * 1 // 4: imsize * 3 // 4] =\
								 counts[:, :, imsize * 1 // 4: imsize * 3 // 4 , imsize * 1 // 4: imsize * 3 // 4]
						else:
							print("Invalid Masking Location")
							exit()
					else:
						full_predictions = probs
						full_counts = counts

				if args.hand_cond:
					full_predictions[ivindices] = 0
					full_counts[ivindices] = 0

				pasted_preds = paste_to_canvas(full_predictions.to(device), fold_op.to(device))
				pasted_counts = paste_to_canvas(full_counts.to(device), fold_op.to(device))
				
				_, _, h, w = pasted_counts.shape

				pasted_preds = pasted_preds[:, :, 0+padding[2]:h-padding[3], 0+padding[0]:w-padding[1]]
				pasted_counts = pasted_counts[:, :, 0+padding[2]:h-padding[3], 0+padding[0]:w-padding[1]]

				pasted_preds_rs = F.interpolate(pasted_preds, size=orig_size[::-1], mode="bilinear")
				pasted_counts_rs = F.interpolate(pasted_counts, size=orig_size[::-1], mode="nearest")

				pred_list.append(pasted_preds_rs)
				count_list.append(pasted_counts_rs)
			
			pasted_preds = torch.sum(torch.stack(pred_list), dim=0)
			pasted_counts = torch.sum(torch.stack(count_list), dim=0)

			final_npimage_ = (pasted_preds / (pasted_counts + 1e-8))[0, 0].cpu().numpy()

			
			final_npimage = np.where(npimg[:, :, 0] == 0, 0, final_npimage_.copy()) # black regions get 0 score

			heatmap_img = cv2.applyColorMap((final_npimage*255.).astype(np.uint8), cv2.COLORMAP_JET)
			fin = cv2.addWeighted(heatmap_img, 0.5, npimg, 0.5, 0)
			cv2.imwrite(f'{out_dir}/color_img_{item_name}.jpg', fin)

			final_image = nparray2img(final_npimage)

			final_image.save(f"{out_dir}/final_{item_name}.png")
			img.save(f"{out_dir}/input_{item_name}.jpg")
	

def main(args):
	model_name = args.model_name
	prefix = ""
	prefix = prefix + f"_{args.ckpt}" if args.ckpt is not None else prefix
	prefix = prefix + f"_{args.wsize}" if args.wsize is not None else prefix 
	# prefix = prefix + f"_nomask" if not args.mask_input else prefix 
	if args.ckpt is None:
		model_path = f"{args.model_dir}/{model_name}_checkpoint.pth"
	else:
		model_path = f"{args.model_dir}/{model_name}_checkpoint_{args.ckpt}.pth"
	
	out_dir = f"{args.out_dir}_benchmark/{model_name}{prefix}"
	os.system(f"mkdir -p {out_dir}")
	
	checkpoint = torch.load(model_path)

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

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

	metadata = json.load(open(args.annotated_frames, "r"))

	if args.split == "val":
		frames = metadata["val_images"]
	elif args.split == "test":
		frames = metadata["test_images"]
	else:
		print("Error: Use a valid split")
		sys.exit()

	infer(out_dir, model, mask_location, frames, args)
		
if __name__ == "__main__":
	GIT_ROOT = get_git_root(__file__)
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
	parser.add_argument('--out_dir', dest='out_dir', type=str,
						default="./inferred_IHdata/epic")
	parser.add_argument('--ckpt', dest='ckpt', type=int,
						default=None)
	parser.add_argument('--annotated_frames', dest='annotated_frames', type=str,
						default=f"{GIT_ROOT}/evaluation/epic-roi/data/annotated_frames.json")
	parser.add_argument('--inp_dir', dest='inp_dir', type=str,
						default=f"{GIT_ROOT}/evaluation/epic-roi/data/roi-data/images")
	parser.add_argument('--split', dest='split', type=str,
						default="val")
	parser.add_argument('--GT_Dir', dest='GT_dir', type=str,
						default=f"{GIT_ROOT}/evaluation/epic-roi/data/roi-data/GT0_cat1234")
	parser.add_argument('--hand_cond', dest='hand_cond', type=str2bool, default=False)
	parser.add_argument('--mask_input', dest='mask_input', type=str2bool, default=True)
	parser.add_argument('--sym_encdec', dest='sym_encdec', type=str2bool, default=False)
	parser.add_argument('--two_heads', dest='two_heads', type=str2bool, default=False)
	parser.add_argument('--loss_masking', dest='loss_masking', type=str2bool, default=False)
	

	args = parser.parse_args()
	main(args)
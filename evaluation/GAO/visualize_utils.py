from PIL import Image, ImageEnhance
import torch
import numpy as np
from torchvision import transforms
import cv2

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
		# print(np.mean(hmap))
		# print(np.mean(hmap))
		# hmap[hmap<0.3] = 0
		# hmap = cv2.GaussianBlur(hmap, (3, 3), 0)
		if blur:
			hmap = cv2.GaussianBlur(hmap, (3, 3), 0)
		# hmap_pooled = ndimage.maximum_filter(hmap, size=15)
		# hmap = np.where(hmap_pooled - hmap < 0.2, hmap, hmap * 0)
		# hmap = hmap - np.min(hmap)
		# hmap = hmap / np.max(hmap)
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


def overlay_colored_heatmaps(uimg, hmaps, colors, blur=True): # (C, 224, 224)
	# post process heatmaps: normalize each channel, blur, threshold
	phmaps = post_process(hmaps, blur) # (C, 224, 224) 

	# generate color map from each heatmap channel
	cmap = generate_color_map(phmaps, colors, blur)

	# generate per-pixel alpha channel and overlay
	alpha = (1-cmap).mean(0)
	overlay = (1-alpha)*uimg + alpha*cmap

	return overlay

def create_vis_fromimgs(img, hm, blur=True):
	imgtensor = to_tensor(img)
	hmtensor = to_tensor(hm)

	overlay = overlay_colored_heatmaps(imgtensor, hmtensor, blur)

	# viz_imgs = [imgtensor, overlay]
	# grid = torchvision.utils.make_grid(viz_imgs, nrow=1, padding=2)
	ol = to_PIL(overlay)

	return img, ol

def create_vis(img_path, hm_path):
	img = Image.open(img_path)
	hm = []
	for hpath in hm_path:
		hmap = np.array(Image.open(hpath)).astype(np.float32)
		hmap = np.where(hmap < 255, hmap * 0, hmap)
		hm.append(hmap)
	hm = np.stack(hm, axis=-1)

	# sel_indices = [0, 3, 8, 15]
	colors = ["red", "green", "blue", "magenta"]

	imgtensor = to_tensor(img)
	hmtensor = torch.from_numpy(hm).permute(2, 0, 1)

	# cropper = transforms.CenterCrop((256, 256))

	# imgtensor = cropper(imgtensor)
	# hmtensor = cropper(hmtensor)

	overlay = overlay_colored_heatmaps(imgtensor, hmtensor, colors, True)

	# viz_imgs = [imgtensor, overlay]
	# grid = torchvision.utils.make_grid(viz_imgs, nrow=1, padding=2)
	ol = to_PIL(overlay)
	img = to_PIL(imgtensor)

	return img, ol
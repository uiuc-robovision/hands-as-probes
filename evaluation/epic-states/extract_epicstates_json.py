import pandas as pd
import pickle, os
import numpy as np
from PIL import Image, ImageChops
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import json

def crop_resize(img, tup, shape=(256, 256)):
	cropped_img = img.crop(tup)
	resized_img = cropped_img.resize(shape, Image.ANTIALIAS)

	return resized_img

def load_pickle(fp):
	with open(fp, "rb") as f:
		data = pickle.load(f)
	return data

def get_path(data_path, row):
	return os.path.join(data_path, row['participant_id'], "object_detection_images", row['video_id'], f"{format(row['frame'], '010d')}.jpg")

def extract_crop(args, row, item):
	bb = item['bb_epicstates']
	oimg_path = get_path(args.data, item)
	oimg = Image.open(oimg_path)
	cpimg = crop_resize(oimg, bb)
	out_path = f"{args.out_path}/{row['split']}/{row['object']}/{row['file']}"
	cpimg.save(f"{args.out_path}/{row['file']}")
	return out_path

def main(args):
	# Load metadata for epic-states annotations
	metadata = pd.read_csv(args.annotations)
	
	# Load Crop data for epic-states
	crop_data = json.load(open(args.crop_data, "r"))
	os.makedirs(args.out_path, exist_ok=True)
	
	output = Parallel(n_jobs=20)(delayed(extract_crop)(args, row, crop_data[row['file']]) \
				for _, row in tqdm(metadata.iterrows(), total=len(metadata)))
	
	print(len(set(output)), len(output))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='frame extraction arguments')
	parser.add_argument('--data', dest='data', type=str,
						default="/home/mohit/EPIC-KITCHENS/")
	parser.add_argument('--crop_data', dest='crop_data', type=str,
						default="./data/epic-states.json")
	parser.add_argument('--annotations', dest="annotations", type=str,
						default="./data/epic-states-annotations.csv")
	parser.add_argument('--out_path', dest='out_path', type=str,
						default="./data/epic-states")
	args = parser.parse_args()
	main(args)
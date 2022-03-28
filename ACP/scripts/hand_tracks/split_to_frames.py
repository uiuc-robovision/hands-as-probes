import os, glob
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from utils.io_utils import load_trajectory, save_trajectory
from absl import app, flags
import multiprocessing
from functools import partial


FLAGS = flags.FLAGS
flags.DEFINE_string('in_dir', '/data01/mohit/Track-Hands/output/original_tracks_0.8_128', '')


def get_trajectory_length(impath):
	images = Image.open(impath)
	exif = images.getexif()
	num_images = exif[0]
	return num_images


def load_trajectory(impath):
	images = Image.open(impath)
	exif = images.getexif()
	num_images = exif[0]
	width = exif[1]
	height = exif[2]
	cols = images.size[0] // exif[1]
	rows = images.size[1] // exif[2]
	all_images = []
	for imnum in range(exif[0]):
		i, j = imnum % cols, imnum // cols
		image = images.crop((i*width, j*height, i*width + width , j*height + height))
		all_images.append(image)
	return all_images

def save_trajectory_as_frames(impath, imlist):
	length = len(imlist)
	os.makedirs(impath, exist_ok=True)
	width = imlist[0].size[0]
	height = imlist[0].size[1]

	for i, new_im in enumerate(imlist):
		ee = new_im.getexif()
		ee[0] = length
		ee[1] = width
		ee[2] = height
		new_im.save(os.path.join(impath, f"{i}.jpg"), exif=ee)

def split_track(outpath, inpath):
	imlist = load_trajectory(inpath)
	outpath_track = os.path.join(outpath, os.path.splitext(os.path.basename(inpath))[0])
	save_trajectory_as_frames(outpath_track, imlist)

def main(_):
	for split in ["train", "validation"]:
		hands_path = os.path.join(FLAGS.in_dir, split, "hand")
		jpgFilenamesList = sorted(glob.glob(f"{hands_path}/*.jpg"))
		print(f"{hands_path}/*.jpg")
		print(len(jpgFilenamesList))
		out_path = os.path.join(FLAGS.in_dir + "_split", split, "hand")
		os.makedirs(out_path, exist_ok=True)
		pool = multiprocessing.Pool(processes=20)
		func = partial(split_track, out_path)

		# pool.map(func, jpgFilenamesList)
		[l for l in tqdm(pool.imap_unordered(func, jpgFilenamesList), total=len(jpgFilenamesList))]

		os.system(f"cp -r {FLAGS.in_dir}/*.pkl {FLAGS.in_dir}_split/")

		# Copy meta information to the target folder
		os.system(f"cp ../annotations/EPIC_55_annotations.csv {FLAGS.in_dir}_split/")
		os.system(f"cp ../annotations/EPIC_test_s1_object_video_list.csv {FLAGS.in_dir}_split/")
		os.system(f"cp ../annotations/EPIC_test_s2_object_video_list.csv {FLAGS.in_dir}_split/")





if __name__ == '__main__':
	app.run(main)

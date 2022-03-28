import os, glob
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from absl import app, flags


FLAGS = flags.FLAGS
flags.DEFINE_string('inp_dir', '/data01/mohit/Track-Hands/output/original_tracks_0.8_128', '')


def get_trajectory_length(impath):
    images = Image.open(impath)
    exif = images.getexif()
    num_images = exif[0]
    return num_images

def parse(path):
	name = os.path.basename(path)
	lst = name.split("_")
	dic = {}
	dic["participant_id"] = lst[0]
	dic["video_id"] = "_".join(lst[0:2])
	dic["start_frame"] = int(lst[2])
	dic["end_frame"] = int(lst[3])
	dic["side"] = lst[4].split(".")[0]

	return dic

def main(_):
	for split in ["train", "validation"]:
		hands_path = os.path.join(FLAGS.inp_dir, split, "hand")
		jpgFilenamesList = sorted(glob.glob(f"{hands_path}/*.jpg"))
		
		print(f"# Tracks in {split} set: {len(jpgFilenamesList)}")
		
		# Create metadata dictionary
		meta_dict = {
			"file": [os.path.basename(i) for i in jpgFilenamesList],
			"participant_id": [parse(i)["participant_id"] for i in jpgFilenamesList],
			"video_id": [parse(i)["video_id"] for i in jpgFilenamesList],
			"start_frame": [parse(i)["start_frame"] for i in jpgFilenamesList],
			"end_frame": [parse(i)["end_frame"] for i in jpgFilenamesList],
			"side": [parse(i)["side"] for i in jpgFilenamesList],
			"length": [get_trajectory_length(i) for i in jpgFilenamesList],
		}

		data = pd.DataFrame(meta_dict)

		data.to_pickle(f"{FLAGS.inp_dir}/annotations_{split}.pkl")




if __name__ == '__main__':
    app.run(main)

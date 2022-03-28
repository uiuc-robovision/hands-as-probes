import os, json
import argparse

def main(args):
	# Make sure the output directory exists
	os.makedirs(args.out_dir, exist_ok=True)

	# Load the old json file
	annot = json.load(open(args.annotated_frames, 'r'))

	frames = annot['val_images'] + annot['test_images']

	for frame in frames:
		frame = frame[:-4] + ".jpg"
		pid = frame[:3]
		vid = frame[:6]
		path = f"{args.data}/{pid}/videos/{vid}_ROI/{frame[7:]}"
		os.system(f"cp {path} {args.out_dir}/{frame}")
		# break

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='extraction arguments')
	parser.add_argument('--data', dest='data', type=str,
						default="/home/mohit/EPIC-KITCHENS")
	parser.add_argument('--out_dir', dest='out_dir', type=str,
						default="./epic-roi-frames")
	parser.add_argument('--annotated_frames', dest='annotated_frames', type=str,
						default="./data/annotated_frames.json")
	parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
	args = parser.parse_args()
	main(args)
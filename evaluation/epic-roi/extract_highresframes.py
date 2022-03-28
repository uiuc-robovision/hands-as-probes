import pickle, os, json, argparse

def main(args):
	metadata = json.load(open(args.annotations, 'r'))
	data_dir = "{}/{}/videos/{}"

	videos = metadata['videos']
	for video in videos:
			path = data_dir.format(args.data, video[:3], video)
			os.makedirs(path + "_ROI", exist_ok=True)
			D = path + ".MP4"
			# print(f"ffmpeg -i {D} -vf scale=1920x1080 -q:v 2 -r 60 {path}_ROI/frame_%010d.jpg")
			os.system(f"ffmpeg -i {D} -vf scale=1920x1080 -q:v 2 -r 60 {path}_ROI/frame_%010d.jpg")



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='training hyper-parameters')
	parser.add_argument('--data', dest='data', type=str,
						default="/home/mohit/EPIC-KITCHENS")
	parser.add_argument('--annotations', dest='annotations', type=str,
						default="./data/annotated_frames.json")
	args = parser.parse_args()
	main(args)

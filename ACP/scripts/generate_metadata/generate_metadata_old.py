import pandas as pd
import pickle
import glob, os, sys
from epic_kitchens.hoa import load_detections, DetectionRenderer
import argparse
import multiprocessing
from functools import partial
import itertools
import random
import pdb

CONTACT_INDICES = [3, 4]
OBSCORE_THRESH = 0.8

def get_best_hands(det, return_handscore=False):
	hands = det.hands
	best_l_score, best_r_score = 0., 0.
	l_state, r_state = None, None
	l_hand, r_hand = None, None
	objects = []
	objects_scores = []
	lh_scores, rh_scores = [], []
	for hand in hands:
		if contact:
			if hand.state.value in CONTACT_INDICES:
				if hand.side.value == 0:
					# Left Hand
					if hand.score > best_l_score:
						best_l_score = hand.score
						l_state = hand.state.value
						l_hand = (hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom)
				elif hand.side.value == 1:
					# Right hand
					if hand.score > best_r_score:
						best_r_score = hand.score
						r_state = hand.state.value
						r_hand = (hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom)
				# objects = [(i.bbox.left, i.bbox.top, i.bbox.right, i.bbox.bottom) for i in det.objects if i.score > OBSCORE_THRESH]
				objects = [(i.bbox.left, i.bbox.top, i.bbox.right, i.bbox.bottom) for i in det.objects]
				objects_scores = [i.score for i in det.objects]
		else:
			if hand.side.value == 0:
				# Left Hand
				if hand.score > best_l_score:
					best_l_score = hand.score
					l_state = hand.state.value
					l_hand = (hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom)
			elif hand.side.value == 1:
				# Right hand
				if hand.score > best_r_score:
					best_r_score = hand.score
					r_state = hand.state.value
					r_hand = (hand.bbox.left, hand.bbox.top, hand.bbox.right, hand.bbox.bottom)
			# objects = [(i.bbox.left, i.bbox.top, i.bbox.right, i.bbox.bottom) for i in det.objects if i.score > OBSCORE_THRESH]
			objects = [(i.bbox.left, i.bbox.top, i.bbox.right, i.bbox.bottom) for i in det.objects]
			objects_scores = [i.score for i in det.objects]
	if return_handscore:
		return l_hand, r_hand, objects, objects_scores, \
				{'l_score': best_l_score, 'r_score': best_r_score, 'l_state': l_state, 'r_state': r_state}
	return l_hand, r_hand, objects, objects_scores

def filter_detections(return_handscore, file):
	vid_dets = load_detections(file)
	lr_handpairs = [get_best_hands(i, return_handscore) for i in vid_dets if len(i.hands) > 0]
	video_id = os.path.splitext(os.path.basename(file))[0]
	part_id = os.path.splitext(os.path.basename(file))[0].split("_")[0]

	video_ids = [video_id for i in vid_dets if len(i.hands) > 0]
	participant_id = [part_id for i in vid_dets if len(i.hands) > 0]

	frames = [i.frame_number for i in vid_dets if len(i.hands) > 0]
	frame_names = [f'frame_{i.frame_number:010d}.jpg' for i in vid_dets if len(i.hands) > 0]

	dic_list = []
	for hand, vid, pid, fname in zip(lr_handpairs, video_ids, participant_id, frame_names):
		dic_list.append((hand, vid, pid, fname))

	return dic_list


def main(args):
	global contact
	contact = args.contact
	detections = args.det_path

	os.makedirs(args.out_path, exist_ok=True)
	
	detection_files = list(sorted(set(glob.glob(f"{detections}/*/*.pkl"))))

	test_s1 = pd.read_csv("../annotations/EPIC_test_s1_object_video_list.csv")
	test_s2 = pd.read_csv("../annotations/EPIC_test_s2_object_video_list.csv")
	to_be_removed_videos = list(test_s1['video_id']) + list(test_s2['video_id'])
	train_videos = list(sorted(set(pd.read_csv("../annotations/EPIC_55_annotations.csv")['video_id']))) + to_be_removed_videos
	
	to_be_removed_parts = ["P01", "P08", "P11", "P02", "P32", "P18", "P04", "P09", "P03"]
	detection_files = [i for i in detection_files if (os.path.splitext(os.path.basename(i))[0] in train_videos)]
	print("Total files", len(detection_files))

	detection_files = [i for i in detection_files if (os.path.basename(i)).split("_")[0] not in to_be_removed_parts]
	print("Total files", len(detection_files))
	
	if args.split_with == "videos":
		# Split Videos
		participant_ids = [os.path.splitext(os.path.basename(f))[0] for f in detection_files]
	elif args.split_with == "participants":
		# Split Pariticpants
		participant_ids = [os.path.splitext(os.path.basename(f))[0].split("_")[0] for f in detection_files]
	else:
		print("Use valid split_wth")
		sys.exit()

	p_ids = sorted(list(set(participant_ids)))

	random.seed(0)
	num_participants = len(p_ids)
	train_size = int(0.95 * num_participants)
	train_ids = random.sample(p_ids, train_size)

	validation_ids = [i for i in p_ids if i not in train_ids]
	pool = multiprocessing.Pool(processes=30)

	func = partial(filter_detections, False)
	func(detection_files[0])
	train_dets = pool.map(func, [j for i, j in zip(participant_ids, detection_files) if i in train_ids])

	if contact:
		train_order = pickle.load(open("../annotations/train_order_reproducibility.pkl", "rb"))
	else:
		train_order = pickle.load(open("../annotations/train_order_nocontact_reproducibility.pkl", "rb"))
	train_dict = {i[0][1]: i for i in train_dets}
	train_dets = [train_dict[vid] for vid in train_order]

	train_dets = list(itertools.chain(*train_dets))

	validation_dets = pool.map(func, [j for i, j in zip(participant_ids, detection_files) if i in validation_ids])

	if contact:
		validation_order = pickle.load(open("../annotations/validation_order_reproducibility.pkl", "rb"))
	else:
		validation_order = pickle.load(open("../annotations/validation_order_nocontact_reproducibility.pkl", "rb"))
	validation_dict = {i[0][1]: i for i in validation_dets}
	validation_dets = [validation_dict[vid] for vid in validation_order]

	validation_dets = list(itertools.chain(*validation_dets))

	ext = ""
	if contact:
		ext += "_contact"

	with open(f"{args.out_path}/train{ext}_{args.split_with}.pkl", "wb") as output_file:
		pickle.dump(train_dets, output_file)

	with open(f"{args.out_path}/validation{ext}_{args.split_with}.pkl", "wb") as output_file:
		pickle.dump(validation_dets, output_file)

if __name__ == "__main__":
	from epic_kitchens.hoa import load_detections
	parser = argparse.ArgumentParser(description='training hyper-parameters')
	parser.add_argument('--det_path', dest='det_path', type=str,
						default="/data01/mohit/InteractionHotspots/detections/hand-objects")
	parser.add_argument('--out_path', dest='out_path', type=str,
						default="./")
	parser.add_argument('--split_with', dest='split_with', type=str, default="videos")
	parser.add_argument('--contact', dest='contact', action="store_true", default=False)
	args = parser.parse_args()
	main(args)

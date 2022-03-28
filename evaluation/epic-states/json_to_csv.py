# Converter
from collections import defaultdict
import json
import pdb
import random
import argparse, os, glob
import itertools
import numpy as np
import pandas as pd
from pathlib import Path


def get_experiment_name(json_file_name):
	base_experiment_name = Path(json_file_name).parent.parent.stem
	experiment_name = base_experiment_name
	suffix = ""
	if "ensemb" in json_file_name:
		suffix += "-ensemb"
	if "ftuneTrue" in json_file_name:
		suffix += "-ftune"
	experiment_name += suffix
	return experiment_name

percentages = ["12.5percent", "25.0percent", "50.0percent", "100.0percent"]

def main(args):
	split = args.split
	dfs_map_allperc = []
	files = sorted(glob.glob(str(Path(args.input_dir) / '**' / f"*_final.json"), recursive=True))
	for perc in percentages:
		filtfiles = [f  for f in files if perc in f]
		dfs_map = []
		dfs_classwise_mean = []
		dfs_classwise_std = []
		for file in filtfiles:
			experiment_name = get_experiment_name(file)
			split_short = "val" if split == "validation" else "test"
			with open(file) as f:
				js = json.load(f)
			map_dict = {f"{k}_{perc}":v for k,v in js.items() if f"{split_short}_mAP" in k}
			df = pd.DataFrame([map_dict], index=[experiment_name])
			dfs_map.append(df)
			classwise_df = pd.DataFrame([js[f"{split_short}_AP_classwise_mean"]], index=[experiment_name])
			dfs_classwise_mean.append(classwise_df)
			classwise_df = pd.DataFrame([js[f"{split_short}_AP_classwise_std"]], index=[experiment_name])
			dfs_classwise_std.append(classwise_df)

		if not len(dfs_map):
			continue
		classwise_mean = pd.concat(dfs_classwise_mean).sort_index()
		classwise_std = pd.concat(dfs_classwise_std).sort_index()
		classwise_mean.to_csv(args.output_dir / f"{split}_classwise_mean_{perc}.csv", float_format='%.3f')
		classwise_std.to_csv(args.output_dir /  f"{split}_classwise_std_{perc}.csv", float_format='%.3f')
		dfs_map_allperc.append(pd.concat(dfs_map))

	final = pd.concat(dfs_map_allperc, axis=1).sort_index()
	final = final[sorted(final.columns, key=lambda c: ["std" not in c,] + [p in c for p in percentages[::-1]], reverse=True)]
	csv_path = args.output_dir / f"{split}_final.csv"
	final.to_csv(csv_path, float_format='%.3f')
	print(f"Saved mAP CSV to {csv_path.absolute()}")
	print(f"- STD across mAP for different evaluation dataset sizes")

	# aggregate mean mAP across seeds and std of each pretraining seed mAP 
	seeds = final[[f for f in final.columns if '_std_' not in f]]
	exp_names = seeds.index
	seeds_per_exp = defaultdict(list)
	for name in exp_names:
		parts = name.split('_sd')
		if parts[0] == 'resnet':
			base_name = 'resnet'
		else:
			base_name = parts[0] + '_' + '_'.join(parts[1].split('_')[1:])
		seeds_per_exp[base_name].append(name)

	data = []
	index = []
	for base_name, exps in seeds_per_exp.items():
		subset = final.loc[exps]
		avgs = subset.mean()[[c for c in final.keys() if '_std_' not in c]]
		stds = subset.std()[[c for c in final.keys() if '_std_' not in c]]
		stds.index = [c.split('_')[0] + '_mAP_std_' + c.split('_mAP_')[-1] for c in stds.index]
		df = pd.concat([avgs, stds])
		data.append(df)
		index.append(base_name)
	
	seeds = pd.DataFrame(data, index)
	csv_path = args.output_dir / f"{split}_final_aggseeds.csv"
	seeds.to_csv(csv_path, float_format='%.3f')
	print(f"Saved mAP CSV to {csv_path.absolute()} (aggregated across seeds)")
	print(f"- STD across mAP for different pretraining seeds")


def aggregate_participant_wise_eval(args):
	split = args.split
	files = sorted(glob.glob(str(Path(args.input_dir) / '**' / f"{split}_participant_avgaps.csv"), recursive=True))
	if len(files) == 0:
		return
	df = pd.concat([pd.read_csv(f, index_col="name") for f in files if Path(f).parent.stem != "results"]).round(3)
	df.sort_index(inplace=True)
	csv_path = args.output_dir / f"{split}_participant_avgaps_all.csv" 
	df.to_csv(csv_path, index=True)
	print(f"Saved aggregated partipant-wise CSV to {csv_path}")


def aggregate_classwise_aps(args):
	for split in ["validation", "test"]:
		files = sorted(glob.glob(str(Path(args.input_dir) / '**' / f"{split}_classwise_aps.csv"), recursive=True))
		if len(files) == 0:
			continue
		df = pd.concat([pd.read_csv(f, index_col="name") for f in files if Path(f).parent.stem != "results"])
		df.sort_index(inplace=True)
		csv_path = args.output_dir / f"{split}_classwise_aps_all.csv" 
		df.to_csv(csv_path, index=True)
		print(f"Saved aggregated class-wise AP CSV to {csv_path}")


def classwise_aps_json(args, anon=False):
	all_files = glob.glob(str(Path(args.input_dir) / '**' / f"*_final.json"), recursive=True)
	for p in ['100.0', '12.5']:
		files = sorted(filter(lambda f: f"{p}percent" in f, all_files))
		random.seed(0)
		random.shuffle(files)
		split = args.split
		split_short = "val" if split == "validation" else "test"
		dfs = []
		for i, file in enumerate(files):
			with open(file) as f:
				js = json.load(f)
			df = pd.DataFrame([js[f"{split_short}_AP_classwise_mean"]], index=[i])
			df["name"] = get_experiment_name(file)
			if not anon:
				df.set_index("name", inplace=True)
				dfs.append(df)
		df = pd.concat(dfs)
		df.sort_index(inplace=True)
		df = df[['open', 'close', 'inhand', 'outofhand', 'raw', 'cooked', 'whole', 'cut', 'peeled', 'unpeeled']]
		csv_path = args.output_dir / f"{split}_classwise_aps_all_{p}percent.csv" 
		df.to_csv(csv_path, index=True)
		print(f"Saved aggregated class-wise AP to {csv_path.absolute()}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='JSON to CSV Converter')
	parser.add_argument('--input_dir', "-d", type=Path, default="./logs_newsplit/EPIC-KITCHENS")
	parser.add_argument('--split', "-s", choices=["validation", "test"], default="validation")
	parser.add_argument('--filter', "-f", default="", type=str, help="Create CSV on certain experiment names only. Comma separated list")
	args = parser.parse_args()

	args.output_dir = args.input_dir / "results"
	args.output_dir.mkdir(exist_ok=True, parents=True)
	args.filter = args.filter.split(",")

	main(args)
	# aggregate_participant_wise_eval(args)
	classwise_aps_json(args)

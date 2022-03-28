import glob, os, json
import pandas as pd
import argparse


def main(args):
	os.makedirs(args.out_dir, exist_ok=True)
	csv_files = sorted(glob.glob(f"{args.inp_dir}/*{args.split}.csv"))
	
	frame_list = []
	for file in csv_files:
		cat = os.path.basename(file).split("_")[1]
		if cat in "1234":
			data = pd.read_csv(file, index_col=0)
			data = data.iloc[:, [0, -1]]
			# data.columns = [i + f" ({metadata['categories'][cat]})" for i in data.columns] # Uncomment for interpretable categories
			data.columns = [i + f" ({cat})" for i in data.columns]
			frame_list.append(data)
	
	final = pd.concat(frame_list, axis=1)
	final.index.name = "Methods"
	final = final.sort_values("Methods")
	if args.verbose:
		print(final.to_string())

	final.to_csv(f"{args.out_dir}/benchmark_{args.split}.csv")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='training hyper-parameters')
	parser.add_argument('--out_dir', dest='out_dir', type=str,
						default="./benchmark_results/")
	parser.add_argument('--inp_dir', dest='inp_dir', type=str,
						default="./results/")
	parser.add_argument('--split', dest='split', type=str,
						default="val")
	parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
	args = parser.parse_args()
	main(args)
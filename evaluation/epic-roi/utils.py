import os, yaml

def load_config(path):
	with open(path, "r") as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	return config


def fileparts(path):
	home = os.path.dirname(path)
	basename = os.path.basename(path)
	basenamenoext, ext = os.path.splitext(basename)
	return home, basename, basenamenoext, ext

def set_numpythreads():
	os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
	os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
	os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
	os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
	os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
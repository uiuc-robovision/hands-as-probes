set -x #echo on

# CUDA_VISIBLE_DEVICES=2 python pysot_tracking.py --random_crops --num_chunks 4 -c 0 &
# CUDA_VISIBLE_DEVICES=3 python pysot_tracking.py --random_crops --num_chunks 4 -c 1 &
# CUDA_VISIBLE_DEVICES=4 python pysot_tracking.py --random_crops --num_chunks 4 -c 2 &
# CUDA_VISIBLE_DEVICES=5 python pysot_tracking.py --random_crops --num_chunks 4 -c 3 &


CUDA_VISIBLE_DEVICES=0 python pysot_tracking.py --maskrcnn --num_chunks 4 -c 0 &
CUDA_VISIBLE_DEVICES=1 python pysot_tracking.py --maskrcnn --num_chunks 4 -c 1 &
CUDA_VISIBLE_DEVICES=2 python pysot_tracking.py --maskrcnn --num_chunks 4 -c 2 &
CUDA_VISIBLE_DEVICES=3 python pysot_tracking.py --maskrcnn --num_chunks 4 -c 3 &

wait

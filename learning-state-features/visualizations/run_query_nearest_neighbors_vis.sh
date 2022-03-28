set -x

python src/visualizations/query_nearest_neighbors.py --data /dev/shm/smodi9/datasets/epic_kitchens/ -g 0 -m /data01/smodi9/VOS/models/resnet/resnet.pth -n 10
python src/visualizations/query_nearest_neighbors.py --data /dev/shm/smodi9/datasets/epic_kitchens/ -g 0 -m /data01/smodi9/VOS/models/ioumf/01-11-2021_tsc/checkpoint_200.pth -n 10
python src/visualizations/query_nearest_neighbors.py --data /dev/shm/smodi9/datasets/epic_kitchens/ -g 0 -m /data01/smodi9/VOS/models/ioumf/07-11-2021_oh_sepmodel_catpe/checkpoint_200.pth -n 10


# Object hand correspondence
# python src/visualizations/query_nearest_neighbors.py --data /dev/shm/smodi9/datasets/epic_kitchens/ -g 0 -m /data01/smodi9/VOS/models/ioumf/07-11-2021_oh_sepmodel_catpe/checkpoint_200.pth -ohm /data01/smodi9/VOS/models/ioumf/07-11-2021_oh_sepmodel_catpe/checkpoint_200.pth -n 5
# python src/visualizations/query_nearest_neighbors.py --data /dev/shm/smodi9/datasets/epic_kitchens/ -g 0 -m /data01/smodi9/VOS/models/ioumf/07-11-2021_oh_sepmodel_catpe/checkpoint_200.pth -ohm /data01/smodi9/VOS/models/ioumf/07-11-2021_oh_sepmodel_catpe/checkpoint_200.pth -n 10 --nearest_hand_embeddings
python visualizations/query_nearest_neighbors.py  -g 2 -m models/partition/ioumf/2022-01-29_ohc_sd0/checkpoint_200.pth -n 10 --nearest_hand_embeddings -ohm models/partition/ioumf/2022-01-29_ohc_sd0/checkpoint_200.pth
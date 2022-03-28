from collections import defaultdict
import os
import random
import shutil
import sys
import argparse
from pathlib import Path
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb
from html4vision import Col, imagetable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torchvision.transforms import functional as F

sys.path.append(str(Path(__file__).absolute().parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
from pretraining.data.epic_kitchen_object_crops import EpicKitchenObjectCrops
from pretraining.model import Resnet, SimCLRLayer, SimCLRLayerMultiHead, SimCLRLayerWithPositionalEncoding


img_transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def crop_track(track_path, idx):
    if idx < 0:
        return Image.open(track_path)
    return Image.open(track_path).crop((128*idx, 0, (idx+1)*128, 128))

class ObjectCrops(EpicKitchenObjectCrops):
    transform = img_transform
    
    def __init__(self, split, hands=False):
        super(ObjectCrops, self).__init__(split, tracks_hand=hands)
        self.hands = hands
        self.index_map = {self.object_crops[i]:i for i in range(len(self.object_crops))}

    def __len__(self):
        return len(self.object_crops)

    def __getitem__(self, item):
        path, idx = self.object_crops[item]
        img = crop_track(path, idx)
        return {"path": str(path), "idx": idx, "image": self.transform(img)}


class BenchmarkingSet(Dataset):
    transform = img_transform
    
    def __init__(self, split):
        super(BenchmarkingSet, self).__init__()
        df = pd.read_csv("/home/smodi9/epic_kitchens/evaluationVOS/Annotations/nov05_2021_fullbatch.csv")
        self.paths = df[df.split==split].path.values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        img = Image.open(path)
        return {"path": str(path), "idx": -1, "image": self.transform(img)}


class Gun71(nn.Module):
    def __init__(self, n_classes, bnorm=False, pretrained=True):
        super(Gun71, self).__init__()
        self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        if bnorm:
            self.head = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, n_classes))
        else:
            self.head = nn.Sequential(
                nn.Linear(512, 512), nn.Dropout(0.0), nn.ReLU(),
                nn.Linear(512, n_classes))

    def forward(self, images):
        return self.trunk(images).reshape(images.shape[0], -1)

class Visualization:
    def __init__(self, args):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        self.num_images = args.num_images
        self.model_path = args.model
        self.device = torch.device(args.gpu)
        self.gun71 = args.model.stem == 'gun71'
        self.output_dir = self.model_path.parent / "vis"
        self.cache_path = self.output_dir / "predictions.pth"
        self.agg_vis_dir = Path("/data01/smodi9/VOS/vis/objects") # change me
        self.benchmarking_set = args.benchmarking_set

        self.oh_corr = args.oh_model is not None
        self.oh_model = args.oh_model
        self.oh_corr_find_nearest_hands = args.nearest_hand_embeddings

        if self.benchmarking_set:
            assert not self.oh_corr
            self.cache_path = self.output_dir / "predictions_bench.pth"

        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

        if self.model_path.parent.stem == 'resnet':
            print("Using pretrained resnet!")
            self.model = Resnet(512).to(self.device)
        elif self.oh_corr and self.gun71:
            print("Gun71!")
            self.cache_path = self.output_dir / "predictions_gun71.pth"
            self.model = Gun71(n_classes=71).to(self.device)
            gun71_model_config = torch.load("/home/mohit/VOS/video-hand-grasps/models/GUN-71/resnet_GUN71_best.pth", map_location=self.device)
            self.model.load_state_dict(gun71_model_config["model_state_dict"])
        elif self.oh_corr:
            print("Object-Hand Correspondence!")
            model_info = torch.load(self.oh_model, map_location=self.device)
            self.model = SimCLRLayerMultiHead(512, 128, num_heads=2).to(self.device)
            self.model.load_state_dict(model_info["model_state_dict"])

            self.model_oh = SimCLRLayerWithPositionalEncoding(512, 128).to(self.device)
            # self.model_oh = SimCLRLayerMultiHead(512, 128, num_heads=1).to(self.device)
            hands_state_dict = {k.split('network.')[1]:v for k,v in model_info["model_object_hands_state_dict"].items() if 'network' in k}
            self.model_oh.network.load_state_dict(hands_state_dict)
            self.model_oh.eval()

            self.agg_vis_dir = Path("/data01/smodi9/VOS/vis/oh")
            if self.oh_corr_find_nearest_hands:
                self.model = self.model_oh
                self.cache_path = self.output_dir / "predictions_hand.pth"
                self.agg_vis_dir = self.agg_vis_dir.parent / "oh_nearest_hands"
        else:
            model_info = torch.load(self.model_path, map_location=self.device)
            if "model_object_hands_state_dict" in model_info:
                print("- With nearest hands")
                num_heads = 2 if 'heads.1.weight' in model_info["model_state_dict"] else 1
                split_feat_ext = model_info['config']['training'].get('split_feat_ext', False)
                self.model = SimCLRLayerMultiHead(512, 128, num_heads, split_feat_ext).to(self.device)
            else:
                self.model = SimCLRLayer(512, 128).to(self.device)
            self.model.load_state_dict(model_info["model_state_dict"])
        
        self.model.eval()
        self.preds = self.get_embeddings()
            
        self.query_paths = []
        self.nearest_neighbors = {}


    def run(self, queries):
        nearest_neighbors = {}
        for query_path, query_index in queries:
            print("\nFinding nearest neighbors for", query_path, query_index)
            self.query_paths.append((query_path, query_index))

            q_img = crop_track(query_path, query_index)
            emb = self.predict(q_img, use_oh_model=self.oh_corr)[0]
            
            paths = self.preds["path"]
            idxs = self.preds["idx"]
            embs = self.preds["embedding"]
            # pdb.set_trace()

            # cost = torch.norm(emb.to(self.device) - embs, dim=1)
            cost = torch.cosine_similarity(emb.unsqueeze(0).to(self.device), embs)
            sorted_idxs = torch.argsort(cost, descending=True)

            max_seen_vids = 1
            max_seen_pids = float('inf')
            i = 0
            min_valid_costs = []
            seen = defaultdict(int)
            if Path(query_path).stem.split("_")[0][0] == 'P':
                seen[Path(query_path).stem.split("_")[0]] += 1
                seen["_".join(Path(query_path).stem.split("_")[:2])] += 1
            else:
                assert Path(query_path).stem.split("_")[1][0] == 'P'
                seen[Path(query_path).stem.split("_")[1]] += 1
                seen["_".join(Path(query_path).stem.split("_")[1:3])] += 1

            while i < len(sorted_idxs) and len(min_valid_costs) < self.num_images:
                index = sorted_idxs[i]
                track = paths[index] 
                if self.oh_corr_find_nearest_hands: 
                    track = track.replace("_hand", "")
                pid = Path(track).stem.split("_")[0]
                vid = "_".join(Path(track).stem.split("_")[:2])
                track_id = Path(track).stem
                if seen[pid] < max_seen_pids and seen[vid] < max_seen_vids:
                    seen[vid] += 1
                    seen[track_id] += 1
                    seen[pid] += 1
                    min_valid_costs.append((track, idxs[index], cost[index].item()))
                i += 1
            
            for p,i,v in min_valid_costs:
                print(f"{p:<50} {i} {v:03f}")

            # save in aggregated directory
            agg_path = self.agg_vis_dir / f"{Path(query_path).parent.stem}_{Path(query_path).stem}"
            agg_path.mkdir(exist_ok=True, parents=True)
            combined_image = Image.fromarray(np.concatenate([crop_track(p, i) for p,i,_ in min_valid_costs], axis=1))
            combined_image.save(agg_path / f"{self.model_path.parent.stem}.jpg")
            if self.oh_corr:
                hand_img = q_img
                obj_img = crop_track(query_path.replace('_hand', ''), query_index)
                q_img = Image.fromarray(np.concatenate([obj_img, hand_img], axis=1))
            q_img.save(agg_path / "query.jpg")

            paths, idxs, costs = zip(*min_valid_costs) 
            nearest_neighbors[(query_path, query_index)] = zip(paths, idxs)
        return nearest_neighbors
        

    def save_html(self, nearest_neighbors, output_name="fridge"):
        image_cache = self.output_dir / "images"
        image_cache.mkdir(exist_ok=True, parents=True)
        cols = [Col('id1', 'ID')]
        for i, (query_path, query_index) in enumerate(nearest_neighbors.keys()):
            query_path = Path(query_path)
            if query_index != -1:
                query = crop_track(query_path, query_index)
                query.save(image_cache / query_path.name)
            else:
                shutil.copyfile(query_path, image_cache / query_path.name)
            nns = []
            for p, idx in nearest_neighbors[(str(query_path), query_index)]: 
                img = crop_track(p, idx)
                local_p = image_cache / f"{Path(p).stem}_{idx}.jpg"
                img.save(local_p)
                nns.append(local_p)
            cols.append(Col("img", f"Query {i}", [(image_cache / query_path.name).relative_to(self.output_dir)] * len(nns)))
            cols.append(Col("img", f"Nearest Neighbor {i}", [x.relative_to(self.output_dir) for x in nns]))

        imagetable(cols, self.output_dir / f'{output_name}.html', f'{self.model_path.parent.stem}',
                    # imscale=1.0,                # scale images to 0.4 of the original size
                    imsize=(128, 128),
                    # sortcol=0,                  #
                    sortable=True,              # enable interactive sorting
                    sticky_header=True,         # keep the header on the top
                    sort_style='materialize',   # use the theme "materialize" from jquery.tablesorter
                    zebra=True,                 # use zebra-striped table
        )
        print(f"\nSaved {output_name}.html to {self.output_dir}")

    def save_grid(self, paths, costs):
        images = [Image.open(p) for p in paths]
        images.append(Image.open(self.query_paths[-1]))
        costs = list(costs)
        costs.append(0)
        out_path = self.output_dir / "vis.png"

        width = images[0].size[0]
        height = images[0].size[1]
        rows = cols = int(np.ceil(np.sqrt(len(images))))
        new_im = Image.new(images[0].mode, (width*cols, height*rows))
        for imnum, im in enumerate(images):
                i, j = imnum % cols, imnum // cols
                new_im.paste(im, (i*width, j*height))
        ee = new_im.getexif()
        ee[0] = len(images)
        ee[1] = width
        ee[2] = height
        return new_im.save(out_path, exif=ee)

    def get_embeddings(self, splits=['train', 'validation']):
        preds = {'path':[], 'idx':[], 'embedding':[]}
        for split in splits:
            path = self.cache_path.parent / f'{self.cache_path.stem}_{split}.pth'
            if path.exists():
                print(split, "reading predictions...")
                predictions = torch.load(path, map_location='cpu')
                print(split, "Done!")
            else:
                print(split, "Generating embeddings...")
                self.cache_path.parent.mkdir(exist_ok=True, parents=True)

                if self.benchmarking_set:
                    dataset = BenchmarkingSet(split)
                else:
                    dataset = ObjectCrops(f'{split}_ioumf', hands=self.oh_corr_find_nearest_hands)
                data = DataLoader(dataset, batch_size=4096, num_workers=16, pin_memory=True) 

                predictions = {"path": [], "idx":[], "embedding":[]}
                for dic in tqdm(data, desc=split, dynamic_ncols=True):
                    embeddings = self.predict(dic["image"])
                    predictions["path"] += dic["path"]
                    predictions["idx"] += dic["idx"]
                    predictions["embedding"].append(embeddings)
                
                predictions["path"] = np.array(predictions["path"])
                predictions["idx"] = np.array(predictions["idx"])
                predictions["embedding"] = torch.cat(predictions["embedding"])
                torch.save(predictions, path)
                print(split, "Generated!")

            [preds[k].append(predictions[k]) for k in predictions]

        preds['path'] = np.concatenate(preds['path'])
        preds['idx'] = np.concatenate(preds['idx'])
        preds["embedding"] = torch.cat(preds["embedding"]).to(self.device)
        return preds

    def predict(self, images, use_oh_model=False):
        if isinstance(images, Image.Image):
            images = ObjectCrops.transform(images).unsqueeze(0)

        images = images.to(self.device)
        with torch.no_grad():
            branch = self.model if isinstance(self.model, Resnet) or isinstance(self.model, Gun71) else self.model.network
            if use_oh_model:
                branch = self.model_oh.network
            embeddings = branch(images).cpu().detach()
        return embeddings


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", type=Path, help="Path to model to visualize")
    ap.add_argument("--gpu", "-g", type=int, default=0, help="GPU to use")
    ap.add_argument("--num_images", "-n", type=int, default=10, help="Number of nearest neighbors")
    ap.add_argument("--output_name", type=str, default='fridge', help="Name of output html")
    ap.add_argument("--oh_model", '-ohm', type=Path, help="Visualize objects that have embeddings close to query hand embedding.")
    ap.add_argument("--nearest_hand_embeddings", action='store_true', help="Visualize objects with hands that close to other hand embeddings.")
    ap.add_argument("--benchmarking_set", action='store_true', help="Find nearest neighbors on benchmarking set.")
    args = ap.parse_args()

    assert ".pth" in str(args.model), "Please specify a checkpoint file or '..../resnet/resnet.pth'"

    args.queries = [
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fork/00023269_P27_06_39391_39661_0_-1_2.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fork/00001210_P22_06_2911_3121_1_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/drawer/00006793_P22_16_77821_78151_0_-2_8.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/egg/00013768_P04_02_46531_47191_1_-1_8.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/knife/00001207_P27_01_8851_8911_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/onion/00018652_P04_02_82321_83281_0_-1_20.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/onion/00010951_P04_05_2071_2311_0_-1_4.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/cupboard/00015588_P22_17_77311_77491_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/fridge/00004569_P04_10_19621_19891_1_-1_4.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/bottle/00020839_P22_08_42481_42541_0_-1_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/egg/00017870_P04_02_15241_15691_1_-1_8.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/drawer/00020672_P04_05_48331_48481_1_-2_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fork/00023660_P22_15_7891_8101_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/drawer/00003895_P22_13_19171_19261_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fridge/00004308_P23_03_21091_21331_0_-1_2.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fridge/00023716_P22_12_1561_1831_1_-1_4.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/knife/00001797_P22_05_52111_52171_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/fridge/00013431_P04_02_10081_10261_1_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/cupboard/00023131_P04_05_96361_96421_0_-1_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fork/00012656_P22_06_1081_1291_0_-2_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fridge/00018223_P22_10_4531_4681_1_-1_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/spoon/00000639_P04_21_6541_6721_1_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/fridge/00019195_P04_13_9721_9901_0_-1_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/cupboard/00019470_P22_17_57121_57181_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/drawer/00000522_P22_07_13741_14071_0_-2_4.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fridge/00016776_P22_15_16531_16771_0_-2_3.jpg', -1),
        
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/onion/00019047_P04_10_19381_19681_1_-2_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/spoon/00014257_P22_09_5041_5371_1_-1_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/potato/00011495_P04_05_31771_32251_0_-2_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/potato/00002999_P04_05_30871_31471_0_-1_12.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/drawer/00012737_P04_02_81721_82111_0_-1_4.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/fridge/00003825_P04_01_37471_37561_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/onion/00004553_P23_04_70891_71161_0_-2_3.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/cupboard/00002200_P27_06_33361_34021_0_-2_12.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/bottle/00010483_P05_03_5011_6001_0_-2_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/fridge/00004923_P22_07_27361_27481_0_-1_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/knife/00010943_P22_10_25831_26041_0_-1_0.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/drawer/00020961_P22_05_53401_53791_0_-2_8.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/drawer/00017777_P27_07_3991_4141_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/spoon/00014617_P22_16_73111_73171_0_-1_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/validation/onion/00017228_P07_10_69421_70141_0_-1_12.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/onion/00022920_P04_02_62341_62551_1_-1_4.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/carrot/00006706_P22_16_37621_37951_0_-2_2.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/egg/00020165_P04_01_42631_42781_0_-2_1.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/train/onion/00008350_P05_08_19051_19291_1_-2_3.jpg', -1),
        ('/home/mohit/VOS/EPIC-2018/evaluationVOS/origclip_generation/sampled_tracks/test/knife/00014922_P04_05_87721_87991_1_-1_3.jpg', -1)
    ]
    
    if args.model.stem == 'gun71' or args.oh_model is not None:
        args.queries = [
            ("/dev/shm/smodi9/datasets/EPIC-KITCHENS/vos_tracks/tracks_partitioned_hand/ioumf/images/P04_05_01_786_0.jpg", 0),
            ("/dev/shm/smodi9/datasets/EPIC-KITCHENS/vos_tracks/tracks_partitioned_hand/ioumf/images/P04_115_01_4_0.jpg", 4),
            ("/dev/shm/smodi9/datasets/EPIC-KITCHENS/vos_tracks/tracks_partitioned_hand/ioumf/images/P04_11_01_117_0.jpg", 3),
            ("/dev/shm/smodi9/datasets/EPIC-KITCHENS/vos_tracks/tracks_partitioned_hand/ioumf/images/P04_06_01_334_0.jpg", 1),
            ("/dev/shm/smodi9/datasets/EPIC-KITCHENS/vos_tracks/tracks_partitioned_hand/ioumf/images/P04_05_01_284_0.jpg", 0),
            ("/dev/shm/smodi9/datasets/EPIC-KITCHENS/vos_tracks/tracks_partitioned_hand/ioumf/images/P04_04_01_301_0.jpg", 2),
        ]

    vis = Visualization(args)
    if not args.benchmarking_set:
        vis.run(args.queries)
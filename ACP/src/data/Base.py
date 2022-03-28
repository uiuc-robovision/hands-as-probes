import os, pickle
import sys
sys.path.append('../')
from helper import set_numpythreads
set_numpythreads()
import logging
import numpy as np
import pandas as pd
from PIL import Image
import imgaug.augmenters as iaa
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, ColorJitter, Normalize, RandomHorizontalFlip, RandomAffine, Resize, RandomApply
from torch.utils.data import DataLoader
from box_utils import BBox
from utils import load_config
from transforms import generate_bbox, generate_bbox_nohands, create_mask, create_validity_mask
from transforms import crop_image, resize_image, generate_bbox_aroundobj_wnegs, generate_bbox_around, generate_bbox_aroundobj

logger = logging.getLogger("main.data")

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    dataset = worker_info.dataset
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    dataset.rng = np.random.default_rng((torch.initial_seed() + worker_id) % np.iinfo(np.int32).max)

class EPICPatchLoader(Dataset):

    def __init__(self, config,
                 split='train',
                 transform=False,
                 length=None):
        # get list of files
        self.rng = np.random.default_rng(0)
        self.data_dir = config['data']['data_dir']
        self.annot_dir = config['data']['annot_dir']
        self.split = split
        self.flips = config['training']['flips']
        self.motionblur = config['training']['motionblur']
        self.negres_fixed = config['training']['negres_fixed']
        self.mask_location = config['training']['mask_location']
        self.obj_mask = config['training']['obj_mask']
        self.hand_mask = config['training']['hand_mask']
        self.obj_sample = config['training']['obj_sample'] if self.obj_mask else False
        self.obj_thresh = config['training']['obj_thresh']
        self.masking = config['training']['masking']
        self.bbshift = config['training']['bbshift']
        self.contact = config['data']['contact']
        self.imsize = config['data']['imsize']
        self.use_no_hands = config['training']['use_no_hands']
        if self.use_no_hands:
            logger.info(f"Not Sampling around Hands")
            # Change Hand Mask to False if set to True
            if self.hand_mask:
                self.hand_mask = False
                logger.info(f"Setting Hand Mask False")

        logger.info(f"Contact set to {self.contact}")
        logger.info(f"Masking set to {self.masking}")
        logger.info(f"Masking Location set to {self.mask_location}")
        logger.info(f"Mask Hand: {self.hand_mask} Obj: {self.obj_mask}")
        logger.info(f"Object Score threshold set to {self.obj_thresh}")
        logger.info(f"Obj Sampling {self.obj_sample}")
        logger.info(f"Resolution of Negative Examples Fixed {self.negres_fixed}")
        logger.info(f"BBSHIFT {self.bbshift}")

        if self.motionblur:
            logger.info("Performing Motion Blur")
            self.aug = iaa.MotionBlur(k=(3, 10), seed=self.rng.integers(0, 100))
        
        ext = "_EPIC55"
        logger.info("Using only EPIC55 data")
        
        if self.contact:
            # file = os.path.join(self.annot_dir, split + "_contact" + f"_benchmarkwobj_videos{ext}_correctsplitfixed.pkl")
            file = os.path.join(self.annot_dir, split + "_contact" + "_videos.pkl")
        else:
            # file = os.path.join(self.annot_dir, split + f"_benchmarkwobj_videos{ext}_correctsplitfixed.pkl")
            file = os.path.join(self.annot_dir, split + "_videos.pkl")
        with open(file, "rb") as f:
            self.meta_info = pickle.load(f)
        self.meta_info = [f for f in self.meta_info if f[0][:2] != (None, None)]

        if length is not None:
            indices = self.rng.choice(len(self.meta_info), length)
            self.meta_info = [self.meta_info[i] for i in indices]

        particips = [i[1] for i in self.meta_info]
        uparts, counts = np.unique(particips, return_counts=True)
        part2count = {p: c for p, c in zip(uparts, counts)}
        self.probs = [1. / part2count[i] for i in particips]

        self.num_images = len(self.meta_info)

        # define transforms
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        s = 1.0
        color_jitter = ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        if transform:
            self.transform = Compose([
                RandomApply([color_jitter], p=0.8),
                ToTensor(),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                normalize
            ])
        else:
            self.transform = Compose([ToTensor(), normalize])

        self.im2tensor = Compose([ToTensor()])
        
        print(f"loaded {split} dataset: {len(self.meta_info)} data points")
        logger.info(f"loaded {split} dataset: {len(self.meta_info)} data points")

    def __len__(self):
        return self.num_images

    def create_bbox_fromhand(self, hand, img):
        if hand is None:
            return None
        return BBox(hand[0], hand[1], hand[2], hand[3], img, True)
        

    def flip_hand_n_boxes(self, img, boxes, horiz=True, vertical=True):
        img = img.copy()
        if horiz:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for ind, bbox in enumerate(boxes):
                if bbox is not None:
                    boxes[ind] = bbox.flip_horizontal()
        if vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            for ind, bbox in enumerate(boxes):
                if bbox is not None:
                    boxes[ind] = bbox.flip_vertical()
            
        return img, boxes
    
    def sample_patches_handsobjects(self, l_bbox, r_bbox, objects):
        outs = None
        ch = self.rng.choice(3) if self.obj_sample else self.rng.choice(2)
        if ch == 0:
            if l_bbox is not None:
                outs = generate_bbox_around(l_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)
            else:
                outs = generate_bbox_around(r_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)
        elif ch == 1:
            if r_bbox is not None:
                outs = generate_bbox_around(r_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)
            else:
                outs = generate_bbox_around(l_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)
        else:
            vobjects = [ob for ob in objects if ob.width > 20 and ob.height > 20]
            if len(vobjects) > 0:
                ob = vobjects[self.rng.choice(len(vobjects))]
                outs = generate_bbox_aroundobj(ob, low_scale=0.5, scale=0.75, fshift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)        

            elif l_bbox is not None:
                outs = generate_bbox_around(l_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)
            else:
                outs = generate_bbox_around(r_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)
        
        return outs
    
    def sample_patches_objects(self, objects):
        outs = None
        vobjects = [ob for ob in objects if ob.width > 20 and ob.height > 20]
        if len(vobjects) > 0:
            ob = vobjects[self.rng.choice(len(vobjects))]
            outs = generate_bbox_aroundobj_wnegs(ob, low_scale=0.5, scale=0.75, fshift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)
        
        return outs

    def __getitem__(self, item):

        outs = None
        while outs is None:
            hand, vid, pid, fname = self.meta_info[item]
            path = f"{self.data_dir}/{pid}/rgb_frames/{vid}/{fname}"
            img = Image.open(path)
            if self.motionblur:
                if self.rng.standard_normal() > 0:
                    img = Image.fromarray(self.aug(image=np.array(img)))
            
            l_hand, r_hand, objects, scores = hand
            objects = [obj for obj, sc in zip(objects, scores) if sc > self.obj_thresh]

            l_bbox = self.create_bbox_fromhand(l_hand, img)
            r_bbox = self.create_bbox_fromhand(r_hand, img)
            objects = [self.create_bbox_fromhand(i, img) for i in objects]
            objects = [ob for ob in objects if ob.width < 150 and ob.height < 150]

            comb_boxes = [l_bbox, r_bbox] + objects

            if self.flips:
                hflips = False
                if self.rng.standard_normal() > 0:
                    hflips = True
                img, comb_boxes = self.flip_hand_n_boxes(img, comb_boxes, hflips, False)
            

            l_bbox = comb_boxes[0]
            r_bbox = comb_boxes[1]
            objects = comb_boxes[2:]

            mask_boxes = []
            if self.obj_mask:
                mask_boxes += comb_boxes[2:] 
            if self.hand_mask:
                mask_boxes += comb_boxes[:2]

            mask = create_mask(img,
                                mask_boxes,# objects
                                )

            seg_mask = Image.fromarray(mask)

            if not self.use_no_hands:
                outs = self.sample_patches_handsobjects(l_bbox, r_bbox, objects)
            else:
                outs = self.sample_patches_objects(objects)

            if outs is None:
                item = self.rng.choice(self.num_images)
                continue


        bbsq, bbaround = outs

        if self.mask_location == "center":
            # Originally patch is at the bottom center, shift the bigger patch down to have masking at the center
            bbaround = bbaround.shift(shiftv=-bbsq.height/2.)

        validity_mask = create_validity_mask(self.imsize, self.masking, mask_location=self.mask_location)

        inp_img = resize_image(crop_image(img, bbaround), self.imsize, self.imsize)
        inp_img_masked = Image.fromarray(np.array(inp_img) * np.expand_dims(validity_mask.astype(np.uint8), -1)) # Hide Hand region

        # seg_image = resize_image(crop_image(img, bbaround), self.imsize, self.imsize)
        seg_image = inp_img_masked.copy()
        seg_mask = resize_image(crop_image(seg_mask, bbsq), self.imsize / 2, self.imsize / 2, type='nn')

        validity_mask_tensor = torch.from_numpy(validity_mask).type(torch.float32).unsqueeze_(0)
        seg_mask_tensor = torch.from_numpy(np.array(seg_mask)/255.).type(torch.float32).unsqueeze_(0)
        
        inp_img_tensor = self.transform(inp_img_masked)
        seg_img_tensor = self.im2tensor(seg_image)

        # torchvision.utils.save_image(inp_img_tensor, f"temp/{item}.png")
        # os.makedirs("temp3", exist_ok=True)
        # inp_img.save(f"temp3/inp_{item}.png")
        # seg_image.save(f"temp3/out_im_{item}.png")
        # seg_mask.save(f"temp3/out_mask_{item}.png")

        return {'img': inp_img_tensor, 'seg_image': seg_img_tensor, "seg_mask": seg_mask_tensor, "valid_mask": validity_mask_tensor}

if __name__ == "__main__":
    config = load_config("../configs/base.yaml")
    print(config)
    data = EPICPatchLoader(split="train",
                                        transform=True,
                                        config=config,
                                        )
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dl = DataLoader(data,
                batch_size=16, shuffle=False,
                num_workers=8,
                worker_init_fn=worker_init_fn,
                drop_last=False)
    # lst = []
    for ind, i in enumerate(dl):
        if ind==5:
            break
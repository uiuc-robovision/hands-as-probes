import os, pickle, lzma
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
from data.Base import EPICPatchLoader
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

class EPICPatchLoaderSeg(EPICPatchLoader):

    def __init__(self, config,
                 split='train',
                 transform=False,
                 length=None
                 ):

        # Check if use_hand_segmask is set to True
        assert config['training']['use_hand_segmask']
        super(EPICPatchLoaderSeg, self).__init__(
                config,
                split=split,
                transform=transform,
                length=length
            )
        
        self.hand_seg_dir = config['data']['hand_seg_dir'] # Location of the hand segmentation masks

    def read_mask(self, path):
        '''
            input
                path: str, takes a string as input
            returns 
                mask: PIL Image
        '''
        mask = Image.fromarray(((pickle.load(lzma.open(path, "rb"))) * 255).astype(np.uint8))
        return mask
    
    def flip_hand_n_boxes(self, img, mask, boxes, horiz=True, vertical=True):
        img = img.copy()
        mask = mask.copy()
        if horiz:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            for ind, bbox in enumerate(boxes):
                if bbox is not None:
                    boxes[ind] = bbox.flip_horizontal()
        if vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            for ind, bbox in enumerate(boxes):
                if bbox is not None:
                    boxes[ind] = bbox.flip_vertical()
            
        return img, mask, boxes
    
    def __getitem__(self, item):

        outs = None
        while outs is None:
            hand, vid, pid, fname = self.meta_info[item]
            path = f"{self.data_dir}/{pid}/rgb_frames/{vid}/{fname}"
            seg_path = f"{self.hand_seg_dir}/{pid}/rgb_frames/{vid}/{fname}" + ".xz"

            img = Image.open(path)
            hand_seg_mask = resize_image(self.read_mask(seg_path), img.size[0], img.size[1], type='nn')

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
                img, hand_seg_mask, comb_boxes = self.flip_hand_n_boxes(img, hand_seg_mask, comb_boxes, hflips, False)

            if not self.use_no_hands:
                outs = self.sample_patches_handsobjects(l_bbox, r_bbox, objects)
            else:
                outs = self.sample_patches_objects(objects)

            if outs is None:
                item = self.rng.choice(self.num_images)
                continue

        
        l_bbox = comb_boxes[0]
        r_bbox = comb_boxes[1]
        objects = comb_boxes[2:]

        mask_boxes = []
        temp_mask = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)

        if self.obj_mask:
            mask_boxes += comb_boxes[2:]
            temp_mask = np.maximum(temp_mask, create_mask(img, comb_boxes[2:]))
        if self.hand_mask:
            mask_boxes += comb_boxes[:2]
            hand_mask = create_mask(img, comb_boxes[:2])
            hand_mask = np.minimum(hand_mask, hand_seg_mask)
            temp_mask = np.maximum(temp_mask, hand_mask)

        seg_mask = Image.fromarray(temp_mask)

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
        # os.makedirs("temp_segold", exist_ok=True)
        # inp_img.save(f"temp_segold/inp_{item}.png")
        # seg_image.save(f"temp_segold/out_im_{item}.png")
        # seg_mask.save(f"temp_segold/out_mask_{item}.png")

        return {'img': inp_img_tensor, 'seg_image': seg_img_tensor, "seg_mask": seg_mask_tensor, "valid_mask": validity_mask_tensor}

if __name__ == "__main__":
    config = load_config("../configs/base.yaml")
    print(config)
    data = EPICPatchLoaderSeg(split="train",
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
        print(i['seg_mask'][0, 0, 32])
        print(i['img'][0, 0, 32])
        if ind==1:
            break

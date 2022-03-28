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
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from box_utils import BBox
from data.Base import EPICPatchLoader
from utils import load_config
from transforms import generate_bbox, generate_bbox_nohands, create_mask, create_validity_mask
from transforms import crop_image, resize_image, save_nparray_asimg, get_rotation_matrix, generate_bbox_around, generate_bbox_aroundobj

logger = logging.getLogger("main.data")

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    dataset = worker_info.dataset
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    dataset.rng = np.random.default_rng((torch.initial_seed() + worker_id) % np.iinfo(np.int32).max)

class EPICPatchLoaderwHands(EPICPatchLoader):

    def __init__(self, config,
                 split='train',
                 transform=False,
                 length=None
                 ):

        super(EPICPatchLoaderwHands, self).__init__(
                config,
                split=split,
                transform=transform,
                length=length
            )

        self.hand_img_size = config['data']['hand_img_size']
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        self.hand_transform = Compose([
            Resize((self.hand_img_size, self.hand_img_size)),
            ToTensor(),
        ])

        if self.use_no_hands:
            logger.info(f"Not Implemented, self.use_no_hands set to {self.use_no_hands}")
            exit()
        
    def sample_patches_handsobjects(self, l_bbox, r_bbox, objects):
        outs = None
        sampled_hand = None
        ispos = None
        ch = self.rng.choice(3) if self.obj_sample else self.rng.choice(2)
        if ch == 0:
            if l_bbox is not None:
                outs, ispos = generate_bbox_around(l_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng, return_pos=True)
                sampled_hand = "l"
            else:
                outs, ispos = generate_bbox_around(r_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng, return_pos=True)
                sampled_hand = "r"
        elif ch == 1:
            if r_bbox is not None:
                outs, ispos = generate_bbox_around(r_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng, return_pos=True)
                sampled_hand = "r"
            else:
                outs, ispos = generate_bbox_around(l_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng, return_pos=True)
                sampled_hand = "l"
        else:
            vobjects = [ob for ob in objects if ob.width > 20 and ob.height > 20]
            if len(vobjects) > 0:
                ob = vobjects[self.rng.choice(len(vobjects))]
                outs = generate_bbox_aroundobj(ob, low_scale=0.5, scale=0.75, fshift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng)        

            elif l_bbox is not None:
                outs, ispos = generate_bbox_around(l_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng, return_pos=True)
                sampled_hand = "l"
            else:
                outs, ispos = generate_bbox_around(r_bbox, low_scale=1.0, scale=1.3, shift=self.bbshift, negres_fixed=self.negres_fixed, rng=self.rng, return_pos=True)
                sampled_hand = "r"
        
        return outs, ispos, sampled_hand


    def __getitem__(self, item):

        outs = None
        sampled_hand = None
        l_hand_img = r_hand_img = None
        while outs is None:
            hand, vid, pid, fname = self.meta_info[item]
            path = f"{self.data_dir}/{pid}/rgb_frames/{vid}/{fname}"
            orig_img = img = Image.open(path)
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

            if l_bbox is not None:
                l_hand_img = crop_image(orig_img, l_bbox.scale(1.2).expand_to_square())
            if r_bbox is not None:
                r_hand_img = crop_image(orig_img, r_bbox.scale(1.2).expand_to_square())

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

            outs, ispos, sampled_hand = self.sample_patches_handsobjects(l_bbox, r_bbox, objects)

            if outs is None:
                item = self.rng.choice(self.num_images)
                continue


        bbsq, bbaround = outs

        if self.mask_location == "center":
            # Originally patch is at the bottom center, shift it up to have masking at the center
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

        hand_sampled = 1
        if sampled_hand == "l":
            hand_img = TF.hflip(l_hand_img)
            if not ispos:
                hand_sampled = 0
        elif sampled_hand == "r":
            hand_img = r_hand_img
            if not ispos:
                hand_sampled = 0
        else:
            hand_img = Image.fromarray(np.zeros((self.hand_img_size, self.hand_img_size, 3)).astype(np.uint8))
            hand_sampled = 0
        
        hand_otensor = self.hand_transform(hand_img)
        hand_tensor = self.normalize(hand_otensor)

        # torchvision.utils.save_image(inp_img_tensor, f"temp/{item}.png")
        # os.makedirs("temp_hands", exist_ok=True)
        # inp_img.save(f"temp_hands/inp_{item}.png")
        # seg_image.save(f"temp_hands/out_im_{item}.png")
        # seg_mask.save(f"temp_hands/out_mask_{item}.png")

        return {'img': inp_img_tensor,
                    'seg_image': seg_img_tensor,
                    "seg_mask": seg_mask_tensor,
                    "valid_mask": validity_mask_tensor,
                    'hand': hand_tensor,
                    'hand_sampled': hand_sampled,
                    'ohand': hand_otensor,
                    }

if __name__ == "__main__":

    config = load_config("../configs/base_hands.yaml")
    print(config)
    data = EPICPatchLoaderwHands(split="train",
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
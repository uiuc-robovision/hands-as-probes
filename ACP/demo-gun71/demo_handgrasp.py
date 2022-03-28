import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
from helper import set_numpythreads
set_numpythreads()

import numpy as np
import pandas as pd
import argparse, pickle
from PIL import Image
from box_utils import BBox

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF

from utils import set_torch_seed, load_config
from model_grasp import ClassificationNet, SimCLRwGUN71
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (10000, 1048576))

def crop_image(img, bbox):
    img_crop = img.crop(bbox.get_bbox())
    return img_crop

def create_bbox_fromhand(hand, img):
        if hand is None:
            return None
        return BBox(hand[0], hand[1], hand[2], hand[3], img, True)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hand_img_size = 128
    
    hand_classes = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10',
       'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20',
       'G21', 'G22', 'G23', 'G24', 'G25', 'G26', 'G27', 'G28', 'G29', 'G30',
       'G31', 'G32', 'G33', 'G34', 'G35', 'G36', 'G37', 'G38', 'G39', 'G40',
       'G41', 'G42', 'G43', 'G44', 'G45', 'G46', 'G47', 'G48', 'G49', 'G51',
       'G52', 'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59', 'G60', 'G62',
       'G63', 'G64', 'G65', 'G66', 'G67', 'G68', 'G69', 'G70', 'G71', 'G72',
       'G73']
    class2id = {j: i for i, j in enumerate(hand_classes)}

    grasp_info = load_config(args.grasp_info)
    sel_classes = grasp_info['easy_classes']['classes']
    sel_names = grasp_info['easy_classes']['names']
    
    filt_ids = []
    filt_names = []
    for c, n in zip(sel_classes, sel_names):
        if c != 'None':
            filt_ids.append(class2id[c])
            filt_names.append(n)

    best_model = torch.load(args.model_path)

    if "gun71_tsc" in os.path.basename(args.model_path).lower():
        hand_model = SimCLRwGUN71(nclasses=len(hand_classes), model_config=best_model["config"]['model']).to(device)
    else:
        hand_model = ClassificationNet(n_classes=len(hand_classes)).to(device)
    
    mweights = best_model["model_state_dict"]
    hand_model.load_state_dict(mweights)

    # Freeze weights and set the model in eval mode
    hand_model.eval()
    for param in hand_model.parameters():
        param.requires_grad = False
    
    img = Image.open("data/P07_10_frame_0000040711.jpg")
    det = pickle.load(open("data/P07_10_frame_0000040711.pkl", "rb"))

    hand, vid, pid, _ = det
    l_hand, r_hand, objects, scores = hand

    l_bbox = create_bbox_fromhand(l_hand, img)
    r_bbox = create_bbox_fromhand(r_hand, img)

    if l_bbox is not None:
        l_hand_img = crop_image(img, l_bbox.scale(1.2).expand_to_square())
    if r_bbox is not None:
        r_hand_img = crop_image(img, r_bbox.scale(1.2).expand_to_square())
    
    hand_transform = TF.Compose([
            TF.Resize((hand_img_size, hand_img_size)),
            TF.ToTensor(),
        ])
    
    normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    
    if l_hand_img is not None:
        l_hand_img = TFF.hflip(l_hand_img)
        l_hand_img.save("data/l_hand.jpg")
        l_hnd = hand_transform(l_hand_img)
        l_tensor = normalize(l_hnd).unsqueeze(0).to(device)
        l_uprobs = hand_model.forward_classifier(l_tensor)[:, filt_ids]
        l_probs = F.softmax(l_uprobs, dim=1).cpu()
        l_grasp_ind = l_probs.argmax(dim=1)[0]
        l_grasp = filt_names[l_grasp_ind]
        print(f"Hand grasp for left hand: {l_grasp}")
    
    if r_hand_img is not None:
        r_hand_img.save("data/r_hand.jpg")
        r_hnd = hand_transform(r_hand_img)
        r_tensor = normalize(r_hnd).unsqueeze(0).to(device)
        r_uprobs = hand_model.forward_classifier(r_tensor)[:, filt_ids]
        r_probs = F.softmax(r_uprobs, dim=1).cpu()
        r_grasp_ind = r_probs.argmax(dim=1)[0]
        r_grasp = filt_names[r_grasp_ind]
        print(f"Hand grasp for right hand: {r_grasp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training hyper-parameters')
    parser.add_argument('--model_path', dest='model_path', type=str,
                        default="../models/GUN-71/GUN71_tsc_seed0_checkpoint_27.pth")
    parser.add_argument('--grasp_info', dest='grasp_info', type=str,
                        default="./metadata/grasp_info.yaml")

    args = parser.parse_args()
    set_torch_seed(0)
    main(args)

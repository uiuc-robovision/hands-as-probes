import torch
import pdb
import math
from torch import overrides
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import utils

class Resnet(nn.Module):
    def __init__(self, emb_size, pretrained=True):
        super(Resnet, self).__init__()
        self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
    
    def forward(self, images):
        return self.trunk(images).reshape(images.shape[0], -1)


class ResnetClassifer(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(ResnetClassifer, self).__init__()
        self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        # for param in self.trunk.parameters():
        #     param.requires_grad = False
        
        self.head = nn.Sequential(
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Linear(512, n_classes),
        )

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        # self.trunk.eval()

    def forward(self, images):
        conv_features = self.trunk(images).reshape(images.shape[0], -1)
        return self.head(conv_features)


class FeatureExtractor(nn.Module):
    def __init__(self, emb_size, base_model="resnet18", pretrained=True, dropout_p=0.4, batchnorm=False, num_heads=None):
        super(FeatureExtractor, self).__init__()
        if base_model == "resnet18":
            self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
            if num_heads is None:
                self.head = nn.Sequential(
                    nn.Linear(512, 512), nn.Dropout(dropout_p), nn.ReLU(),
                    nn.Linear(512, emb_size), nn.ReLU())
            else:
                self.head = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(512, 512), nn.Dropout(dropout_p), nn.ReLU(),
                        nn.Linear(512, emb_size), nn.ReLU()
                    ) for _ in range(num_heads)
                ])
        elif base_model == "resnet50":
            self.trunk = nn.Sequential(*list(models.resnet50(pretrained=pretrained).children())[:-1])
            self.head = nn.Sequential(
                nn.Linear(2048, 1024), nn.Dropout(dropout_p), nn.ReLU(),
                nn.Linear(1024, emb_size), nn.ReLU())

    def forward(self, images, head_idx=None):
        conv_features = self.trunk(images).reshape(images.shape[0], -1)
        if head_idx is None and isinstance(self.head, nn.ModuleList) and len(self.head) == 1:
            head_idx = 0
        if head_idx is None:
            return self.head(conv_features)
        return self.head[head_idx](conv_features)


# https://github.com/kekeblom/tcn/blob/master/tcn.py
def normalize(x):
    buffer = torch.pow(x, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    normalization_constant = torch.sqrt(normp)
    output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
    return output


class SimCLRLayer(nn.Module):
    def __init__(self, embsz, loss_embsz, freeze_resnet=False, normalize=False, alpha=1):
        super(SimCLRLayer, self).__init__()
        self.emb_size = embsz
        self.network = FeatureExtractor(self.emb_size)
        self.lossemb = nn.Linear(self.emb_size, loss_embsz)
        self.freeze_resnet = freeze_resnet
        self.normalize = normalize
        self.alpha = alpha
        if self.freeze_resnet:
            for param in self.network.trunk.parameters():
                param.requires_grad = False

    def forward(self, imgs, head_idx=None):
        out = self.lossemb(self.network(imgs))
        if self.normalize:
            out = normalize(out) * self.alpha
        return out

    def set(self, train=True):
        if train:
            self.train()
            if self.freeze_resnet:
                self.network.trunk.eval()
        else:
            self.eval()


class SimCLRLayerMultiHead(nn.Module):
    def __init__(self, embsz, loss_embsz, num_heads=1, use_feat_ext_heads=False):
        super(SimCLRLayerMultiHead, self).__init__()
        self.emb_size = embsz
        self.use_feat_ext_heads = use_feat_ext_heads
        num_heads_feat_ext = num_heads if use_feat_ext_heads else 1
        self.network = FeatureExtractor(self.emb_size, num_heads=num_heads_feat_ext)
        self.heads = nn.ModuleList([nn.Linear(self.emb_size, loss_embsz) for _ in range(num_heads)])

    def forward(self, imgs, head_idx=0):
        head_idx_feat_ext = head_idx if self.use_feat_ext_heads else 0
        return self.heads[head_idx](self.network(imgs, head_idx=head_idx_feat_ext))

    def set(self, train=True):
        if train:
            self.train()
        else:
            self.eval()


class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features=4, num_frequencies=12):
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def forward(self, coords):
        # [bs * n_hands, (posx, posy, scalex, scaley)]
        bs = coords.shape[0]
        coords = coords.reshape(-1, self.in_features)
        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]
                sin = torch.unsqueeze(torch.sin((2 ** i) * math.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * math.pi * c), -1)
                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(bs, -1, self.out_dim)


class SimCLRLayerWithPositionalEncoding(nn.Module):
    def __init__(self, embsz, loss_embsz, use_feat_ext_heads=False, num_objects=3, pe_infeatures=4, only_pe=False):
        super(SimCLRLayerWithPositionalEncoding, self).__init__()
        self.emb_size = embsz
        self.use_feat_ext_heads = use_feat_ext_heads
        self.num_objects = num_objects
        self.only_pe = only_pe
        self.network = FeatureExtractor(self.emb_size)
        self.pe = PosEncodingNeRF(pe_infeatures)
        self.pe_head = nn.Sequential(
            nn.Linear(self.pe.out_dim, self.emb_size), nn.Dropout(0.4), nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size),
        )
        self.head =  nn.Sequential(
                nn.Linear(self.emb_size*(num_objects+int(not only_pe)), self.emb_size), nn.Dropout(0.4), nn.ReLU(),
        )
                
        self.lossemb = nn.Linear(self.emb_size, loss_embsz)

    def set(self, train=True):
        if train:
            self.train()
        else:
            self.eval()

    def forward(self, imgs, positions=None, head_idx=0):
        assert head_idx == 0
        if not self.only_pe:
            img_feats = self.network(imgs)
        if positions is None:
            return self.lossemb(img_feats)
        
        pe_feats = self.pe(positions)
        pe_feats = [self.pe_head(pe_feats[:, i]) for i in range(self.num_objects)]
        if self.only_pe:
            final_feats = torch.cat([*pe_feats], dim=-1)
        else:
            final_feats = torch.cat([img_feats, *pe_feats], dim=-1)
        return self.lossemb(self.head(final_feats))


class ActionClassifier(nn.Module):
    def __init__(self, config, classes):
        super(ActionClassifier, self).__init__()
        self.num_classes = len(classes)
        self.feat_size = config["model"]["emb_size"]

        self.network = FeatureExtractor(self.feat_size)
        self.head = nn.Linear(512*2, self.num_classes)

    def forward(self, inp):
        im1, im2 = inp['img1'], inp['img2']

        features1 = self.extractor(im1)
        features2 = self.extractor(im2)
        features_stacked = torch.cat([features1, features2], dim=1)

        probs = self.classifier(features_stacked)

        return {'probs': probs}

class StateClassifierwSimCLR(nn.Module):

    def __init__(self, model_config, loss_embsz):
        super(StateClassifierwSimCLR, self).__init__()
        self.emb_size = model_config["emb_size"]
        self.num_classes = model_config["num_classes"]
        self.network = FeatureExtractor(self.emb_size, model_config.get("backbone", "resnet18"),
                                        model_config.get("pretrained", True), model_config.get("dropout", 0.4))
        self.classifier = nn.Linear(self.emb_size, self.num_classes)
        self.lossemb = nn.Linear(self.emb_size, loss_embsz)

    def forward_classifier(self, imgs):
        return self.classifier(self.network(imgs))

    def forward_simclr(self, imgs):
        return self.lossemb(self.network(imgs))

class ABMIL(nn.Module):
    MAX_TRACK_LEN = 16
    def __init__(self, model_config, device):
        super(ABMIL, self).__init__()
        self.num_classes = model_config["abmil_classes"]
        self.emb_size = model_config["emb_size"]
        self.network = FeatureExtractor(self.emb_size)
        self.classifier1 = nn.Linear(self.emb_size, self.num_classes)
        self.classifier2 = nn.Linear(self.emb_size, self.num_classes)
        self.idxs = torch.arange(self.MAX_TRACK_LEN).to(device).unsqueeze(0)

    def get_features(self, images):
        """
        Input: [bs, MAX_TRACK_LENGTH, 3, height, width]
        Output: [MAX_TRACK_LENGTH * bs, emb_size]
        """
        bs = images.shape[0]
        features = self.network(images.reshape(bs * images.shape[1], *images.shape[2:]))
        return features.reshape(bs, -1, self.emb_size)

    def get_logits(self, images, num, state="before"):
        bs = len(num)
        features = self.get_features(images)
        if state == "before":
            logits = self.classifier1(features.reshape(-1, self.emb_size)).reshape(bs, -1, self.num_classes)
        elif state == "after":
            logits = self.classifier2(features.reshape(-1, self.emb_size)).reshape(bs, -1, self.num_classes)
        else:
            raise NotImplementedError
        logits[num.unsqueeze(1) <= self.idxs] = 0.0 #float("-inf")
        return logits

    def forward(self, before_images, after_images, before_num, after_num):
        """
        Input: [bs, MAX_TRACK_LENGTH, 3, height, width], [bs]
        Output [num_states, num_objects, batch_size]
        """
        before_logits = self.get_logits(before_images, before_num, "before")
        after_logits = self.get_logits(after_images, after_num, "after")
        return before_logits.sum(dim=1) / before_num.unsqueeze(1), after_logits.sum(dim=1) / after_num.unsqueeze(1)

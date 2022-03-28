import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):

    def __init__(self, feat_size, bnorm=False, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        if bnorm:
            self.head = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, feat_size))
        else:
            self.head = nn.Sequential(
                nn.Linear(512, 512), nn.Dropout(0.0), nn.ReLU(),
                nn.Linear(512, feat_size))

    def forward(self, images):
        images = images.clone()
        conv_features = self.trunk(images).reshape(images.shape[0], -1)

        feats = self.head(conv_features)

        return feats

class ClassificationNet(nn.Module):

    def __init__(self, n_classes, bnorm=False, pretrained=True):
        super(ClassificationNet, self).__init__()
        self.trunk = nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-1])
        if bnorm:
            self.head = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                nn.Linear(512, n_classes))
        else:
            self.head = nn.Sequential(
                nn.Linear(512, 512), nn.Dropout(0.0), nn.ReLU(),
                nn.Linear(512, n_classes))

    def forward(self, dict):
        images = dict['img']
        conv_features = self.trunk(images).reshape(images.shape[0], -1)

        feats = self.head(conv_features)

        return feats
        
    def forward_classifier(self, imgs):
        conv_features = self.trunk(imgs).reshape(imgs.shape[0], -1)

        feats = self.head(conv_features)

        return feats

class SimCLRwGUN71(nn.Module):

    def __init__(self, nclasses=71, model_config=None):
        super(SimCLRwGUN71, self).__init__()
        self.emb_size = model_config["emb_size"]
        self.extractor = FeatureExtractor(self.emb_size, bnorm=model_config.get('headbnorm', False))
        self.lossemb = nn.Linear(self.emb_size, model_config["loss_emb_size"])
        self.classifier = nn.Linear(self.emb_size, nclasses)

    def forward_embedding(self, imgs):
        return self.lossemb(self.extractor(imgs))
    
    def forward_classifier(self, imgs):
        return self.classifier(self.extractor(imgs))


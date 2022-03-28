import torch
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn as nn

class SegmentationNetDeeper(nn.Module):

    def __init__(self, model_config):
        super(SegmentationNetDeeper, self).__init__()
        bckbone = model_config.get("backbone", "resnet18")
        
        if bckbone == "resnet18":
            self.extractor = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
            self.channels = 512
        elif bckbone == "resnet50":
            self.extractor = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
            self.channels = 2048
        elif bckbone == "resnet101":
            self.extractor = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-2])
            self.channels = 2048
        
        self.segmentation_head = nn.Sequential(
                    nn.ConvTranspose2d(self.channels, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 1, 4, stride=2, padding=1),
                    )
        
        if model_config.get("avg_output", None) is not None:
            self.segmentation_head.add_module("avg_pool2d_1", nn.AvgPool2d((5, 5), padding=2, stride=1))

    def forward(self, inp):
        feats = self.extractor(inp)

        probs = self.segmentation_head(feats)

        return {'pred': probs}

    def infer(self, inp):
        feats = self.extractor(inp)

        probs = self.segmentation_head(feats)

        probs = torch.sigmoid(probs)
        temp = torch.abs(probs - 0.5).mean(-1).mean(-1)
        ind = torch.max(temp, dim=1)[1]
        probs = probs[:, ind]

        return {'pred': probs}

    def infer_cshifted(self, inp):
        feats = self.extractor(inp)

        probs = self.segmentation_head(feats)
        probs = torch.sigmoid(probs)
        probs = torch.max(probs, dim=1, keepdim=True)[0]

        return {'pred': probs}


class SegmentationNetDeeperBig(SegmentationNetDeeper):

    def __init__(self, model_config):
        super(SegmentationNetDeeperBig, self).__init__(model_config)
        
        self.segmentation_head = nn.Sequential(
                    nn.ConvTranspose2d(self.channels, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 1, 4, stride=2, padding=1),
                    )

        if model_config.get("avg_output", False):
            self.segmentation_head.add_module("avg_pool2d_1", nn.AvgPool2d((5, 5), padding=2, stride=1))

class SegmentationNetDeeperSeg(nn.Module):

    def __init__(self, model_config):
        super(SegmentationNetDeeperSeg, self).__init__()
        self.extractor = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])

        self.segmentation_head = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=3),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 1, 4, stride=2, padding=1),
                )

    def forward(self, inp):
        feats = self.extractor(inp)

        probs = self.segmentation_head(feats)

        return {'pred': probs}

class SegmentationNetDeeperTwohead(nn.Module):

    def __init__(self, model_config, emb_size=20):
        super(SegmentationNetDeeperTwohead, self).__init__()
        bckbone = model_config.get("backbone", "resnet18")

        if bckbone == "resnet18":
            self.extractor = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
            self.channels = 512
        elif bckbone == "resnet50":
            self.extractor = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
            self.channels = 2048
        elif bckbone == "resnet101":
            self.extractor = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-2])
            self.channels = 2048
        
        self.segmentation_head = nn.Sequential(
                    nn.ConvTranspose2d(self.channels, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 1, 4, stride=2, padding=1),
                    )
        self.classifier = nn.Sequential(
                        nn.Linear(self.channels, 512),
                        nn.ReLU(),
                        nn.Linear(512, emb_size))
        
        self.emb_size = emb_size

        if model_config.get("avg_output", None) is not None:
            self.segmentation_head.add_module("avg_pool2d_1", nn.AvgPool2d((5, 5), padding=2, stride=1))

    def forward(self, inp):
        feats = self.extractor(inp)

        avg_feats = feats.mean(dim=-1).mean(dim=-1)

        hand_emb = self.classifier(avg_feats)

        probs = self.segmentation_head(feats)

        return {'pred': probs, 'hand_emb': hand_emb}

    def infer_cshifted(self, inp):
        feats = self.extractor(inp)

        avg_feats = feats.mean(dim=-1).mean(dim=-1)

        hand_emb = self.classifier(avg_feats)

        probs = self.segmentation_head(feats)

        probs = torch.sigmoid(probs)
        probs = torch.max(probs, dim=1, keepdim=True)[0]

        return {'pred': probs, 'hand_emb': hand_emb}
    
    def infer_cshiftedgrasp(self, inp):
        feats = self.extractor(inp)
        avg_feats = feats.mean(dim=-1).mean(dim=-1)
        hand_emb = self.classifier(avg_feats)

        return {'hand_emb': hand_emb}

class SegmentationNetDeeperTwoheadBig(SegmentationNetDeeperTwohead):

    def __init__(self, model_config, emb_size=20):
        super(SegmentationNetDeeperTwoheadBig, self).__init__(model_config, emb_size=emb_size)
        
        self.segmentation_head = nn.Sequential(
                    nn.ConvTranspose2d(self.channels, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 1, 4, stride=2, padding=1),
                    )
        
        if model_config.get("avg_output", False):
            self.segmentation_head.add_module("avg_pool2d_1", nn.AvgPool2d((5, 5), padding=2, stride=1))
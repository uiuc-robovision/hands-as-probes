import torch
import pdb
from torchvision.transforms.functional import to_pil_image
from PIL import Image

from .epic_tracks import EpicTracks
from .epic_kitchen_object_crops import EpicKitchenObjectCrops
from .transforms import get_simclr_transform


class EpicTracks4VanillaSimCLR(EpicKitchenObjectCrops):
    def __init__(self, split='train', iou_threshold=1.0):
        super(EpicTracks4VanillaSimCLR, self).__init__(split, iou_threshold)
        self.simclr_transform = get_simclr_transform(s=1)

    def __getitem__(self, item):
        path, idx = self.object_crops[item]
        image = Image.open(path).crop((128*idx, 0, (idx+1)*128, 128))
        image1 = self.simclr_transform(image) 
        image2 = self.simclr_transform(image)
        return image1, image2

class EpicTracks4SimCLR(EpicTracks):
    def __init__(self, split='train', subset=None, segment_len_range=(10, 32)):
        super(EpicTracks4SimCLR, self).__init__(split=split, subset=subset, segment_len_range=segment_len_range)
        self.simclr_transform = get_simclr_transform(s=1)

    def __getitem__(self, item):
        track_path = self.track_files[item]
        start_idx, end_idx = self.extract_segment_from_track(track_path)
        segment_len = end_idx - start_idx
        window = segment_len // 4

        x, y = ((torch.rand(2) * window).long() +
                (torch.rand(1) * (segment_len - window + 1)).long()).tolist()
        x += start_idx
        y += start_idx
        frame1 = self.simclr_transform(self.extract_image_from_track(item, x))
        frame2 = self.simclr_transform(self.extract_image_from_track(item, y))
        return {"pos1": frame1, "pos2": frame2}

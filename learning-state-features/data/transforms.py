import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

from .data_utils import IM_DIM

def get_mitstates_transform(augment, size=128):
    if augment:
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        transform = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(0.7, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomApply([color_jitter], p=0.8),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    return transform

def unnormalize(img):
    return img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

def unnormalize_tensor(img):
    return img * torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)

def get_simclr_transform(s=1, allow_hflip=True):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=128, scale=(0.5, 1.0)),
                                          transforms.RandomHorizontalFlip() if allow_hflip else lambda x: x, 
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=int(0.1 * 128) + 1, sigma=(0.1, 2.0)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return data_transforms

class SimClrTransform:
    def __init__(self, deterministic=False, allow_hflip=True, s=1) -> None:
        if not deterministic:
            self.transform = get_simclr_transform(s, allow_hflip=allow_hflip)
            return
        
        cj = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.color_jitter_params = cj.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)
        self.final_transforms = transforms.Compose([
                                    transforms.GaussianBlur(kernel_size=int(0.1 * 128) + 1, sigma=(0.1, 2.0)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.rrc = transforms.RandomResizedCrop(size=128)
        self.rrc_params = None
        self.enable_flip = torch.rand(1) < 0.5 if allow_hflip else False
        self.enable_jitter = torch.rand(1) < 0.8
        self.enable_grayscale = torch.rand(1) < 0.2
        self.transform = self.deterministic_transform
            
    def deterministic_transform(self, img):
        if self.rrc_params is None:
             # get RandomResizedCrop parameters. Default ratio is provided.
            self.rrc_params = self.rrc.get_params(img, scale=(0.5, 1.0), ratio=(0.75, 4/3))
        img = TF.resized_crop(img, *self.rrc_params, size=(128, 128))
        if self.enable_flip: 
            img = TF.hflip(img)
        if self.enable_jitter:
            for fn_id in self.color_jitter_params[0]:
                if fn_id == 0:
                    img = TF.adjust_brightness(img, self.color_jitter_params[1])
                elif fn_id == 1:
                    img = TF.adjust_contrast(img, self.color_jitter_params[2])
                elif fn_id == 2:
                    img = TF.adjust_saturation(img, self.color_jitter_params[3])
                elif fn_id == 3:
                    img = TF.adjust_hue(img, self.color_jitter_params[4])
        if self.enable_grayscale: 
            img = TF.to_grayscale(img, num_output_channels=3)
        img = self.final_transforms(img)
        assert img.shape[1] == 128 and img.shape[2] == 128, img.shape
        return img

    def apply(self, img):
        return self.transform(img)

    def __call__(self, img):
        return self.apply(img)


def get_track_transform(transform_list, max_track_len=None):
    T = []
    if "coloruni" in transform_list:
        T.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))
    T += [transforms.ToTensor(), ReshapeTrack(IM_DIM)]
    if "temporal" in transform_list:
        T.append(TemporalSegment(max_track_len))
    if "crop" in transform_list:
        T.append(TensorJitter(color="colorrand" in transform_list, crop="crop" in transform_list, blur="blur" in transform_list))
    if "flip" in transform_list:
        T.append(BatchRandomFlip())
    if "pad" in transform_list:
        T.append(PadToLength(max_track_len))
    T.append(TensorNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(T)


class PadToLength(object):
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, imgtensor):
        track_len = imgtensor.shape[0]
        return torch.cat((imgtensor, torch.zeros(self.max_length - track_len, 3, IM_DIM, IM_DIM)), dim=0)


class TensorJitter(object):
    def __init__(self, color=True, crop=False, blur=False):
        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1
        T = []
        if color:
            T.append(transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8))
        if crop:
            T.append(transforms.RandomResizedCrop(size=128, scale=(0.5, 1.0), interpolation=Image.BICUBIC))
        if blur:
            T.append(transforms.RandomApply([transforms.GaussianBlur(int(0.1 * 128) + 1, sigma=(0.1, 2.0))], p=0.5))
        self.transform = transforms.Compose(T)

    def __call__(self, inp):
        for i in range(inp.shape[0]):
            inp[i] = self.transform(inp[i])
        return inp


class ReshapeTrack(object):
    def __init__(self, imdim=128):
        self.imdim = imdim

    def __call__(self, img):
        track_len = img.shape[-1] // self.imdim
        return img.reshape(3, self.imdim, track_len, self.imdim).permute(2, 0, 1, 3)


class BatchRandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        return TF.hflip(img) if torch.rand(1) < self.prob else img


class TemporalSample(object):
    def __init__(self, max_track_length):
        self.max_track_length = max_track_length

    def __call__(self, track):
        window = max((torch.rand(1) * 4 + 1).long().item(), (track.shape[0] - 1) // self.max_track_length + 1, 2)
        num_frames = track.shape[0] // window
        idxs = (torch.rand(num_frames) * window).long() + torch.arange(num_frames).long() * window
        track = track[idxs]
        ends_idx = (torch.rand(2) * 2 + 1).long()
        return track if track.shape[0] <= 8 else track[ends_idx[0]:-ends_idx[1]]


class TemporalSegment(object):
    def __init__(self, max_track_length):
        self.max_track_length = max_track_length

    def __call__(self, track):
        track_len = track.shape[0]
        if track_len <= self.max_track_length:
            return track
        idx = torch.randperm(track_len)[:self.max_track_length].sort()[0]
        return track[idx]


class TensorNormalize(object):
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor


class ToPILTrack(object):
    def __init__(self, imdim=128):
        self.image2tensor = transforms.ToTensor()
        self.tensor2image = transforms.ToPILImage()
        self.imdim = imdim

    def __call__(self, img):
        typ = img.getexif().get(0, None)
        if typ is None:
            return img
        timg = self.image2tensor(img)
        timg = timg.reshape(3, timg.shape[1] // 128, 128, timg.shape[2] // 128, 128)
        timg = timg.permute(0, 2, 1, 3, 4).reshape(3, 128, -1, 128)[:, :, :typ, :].reshape(3, 128, -1)
        return self.tensor2image(timg)

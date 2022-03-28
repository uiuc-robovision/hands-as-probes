import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

IM_DIM = 128
MAX_TRACK_LEN = 20


def get_simclr_transform(s=1, imsize=128, alwaysflip=False):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    T = []
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    if alwaysflip:
        T.append(Flip())
    T = T + [transforms.RandomResizedCrop(size=imsize, scale=(0.5, 1.0)),
                                        #   transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.GaussianBlur(kernel_size=int(0.1 * 128) + 1, sigma=(0.1, 2.0)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    data_transforms = transforms.Compose(T)
    return data_transforms

def get_track_transform(transform_list, imsize=128):
    T = []
    if "color" in transform_list:
        T.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))
    T += [transforms.ToTensor(), ReshapeTrack(imsize)]
    if "alwaysflip" in transform_list:
        T.append(Flip())
    if "temporal" in transform_list:
        T.append(TemporalSegment(MAX_TRACK_LEN))
    if "crop" in transform_list:
        T.append(TensorJitter(color=False, crop="crop" in transform_list, rotate="rotate" in transform_list, size=imsize))
    # if "flip" in transform_list:
    #     T.append(BatchRandomFlip())
    if "pad" in transform_list:
        T.append(PadToLength(MAX_TRACK_LEN))
    T.append(TensorNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(T)


class PadToLength(object):
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, imgtensor):
        track_len = imgtensor.shape[0]
        return torch.cat((imgtensor, torch.zeros(self.max_length - track_len, 3, IM_DIM, IM_DIM)), dim=0)


class TensorJitter(object):
    def __init__(self, color=True, crop=False, rotate=False, blur=False, size=128):
        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1
        self.to_pil = transforms.ToPILImage(mode='RGB')
        T = []
        if crop:
            T.append(transforms.RandomResizedCrop(size=size, scale=(0.5, 1.0), interpolation=Image.BICUBIC))
        if color:
            T.append(transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8))
        if blur:
            T.append(transforms.RandomApply([transforms.GaussianBlur(9)], p=0.5))
        if rotate:
            T.append(transforms.RandomAffine(30))
        T.append(transforms.ToTensor())
        self.transform = transforms.Compose(T)

    def __call__(self, inp):
        for i in range(inp.shape[0]):
            temp = self.to_pil(inp[i])
            inp[i] = self.transform(temp)
        return inp


class ReshapeTrack(object):
    def __init__(self, imdim=128):
        self.imdim = imdim

    def __call__(self, img):
        track_len = img.shape[-1] // self.imdim
        return img.reshape(3, self.imdim, track_len, self.imdim).permute(2, 0, 1, 3)

class Flip(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return TF.hflip(img)


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

import math
from PIL import Image


class RandomResizedCropSeg(torch.nn.Module):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=transforms.InterpolationMode.BILINEAR):
        super().__init__()
        self.size = (int(size), int(size))
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = TF._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img1, img2):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img1, self.scale, self.ratio)
        return TF.resized_crop(img1, i, j, h, w, self.size, self.interpolation), TF.resized_crop(img2, i, j, h, w, self.size, transforms.InterpolationMode.NEAREST) 

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
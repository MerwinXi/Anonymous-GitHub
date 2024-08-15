import random
import cv2
import numpy as np
import torch
import numbers
import collections
from PIL import Image


def to_tensor(data):
    """Convert various types of data to a torch.Tensor.

    Supported types are: numpy.ndarray, torch.Tensor, Sequence, int, and float.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: The converted data.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'Type {type(data)} cannot be converted to tensor.')


class ToTensor:
    """Convert specified keys in a sample to torch.Tensor.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys=None, cfg=None):
        self.keys = keys or ['img', 'mask']

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in self.keys:
            if key in ['img_metas', 'gt_masks', 'lane_line']:
                data[key] = sample[key]
            else:
                data[key] = to_tensor(sample[key])
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(keys={self.keys})'


class RandomLROffsetLABEL:
    """Randomly offset the image and label horizontally."""

    def __init__(self, max_offset, cfg=None):
        self.max_offset = max_offset

    def __call__(self, sample):
        img = np.array(sample['img'])
        label = np.array(sample['mask'])
        offset = np.random.randint(-self.max_offset, self.max_offset)
        h, w = img.shape[:2]

        if offset > 0:
            img[:, offset:] = img[:, :w - offset]
            img[:, :offset] = 0
            label[:, offset:] = label[:, :w - offset]
            label[:, :offset] = 0
        elif offset < 0:
            real_offset = -offset
            img[:, :w - real_offset] = img[:, real_offset:]
            img[:, w - real_offset:] = 0
            label[:, :w - real_offset] = label[:, real_offset:]
            label[:, w - real_offset:] = 0

        sample['img'] = img
        sample['mask'] = label
        return sample


class RandomUDOffsetLABEL:
    """Randomly offset the image and label vertically."""

    def __init__(self, max_offset, cfg=None):
        self.max_offset = max_offset

    def __call__(self, sample):
        img = np.array(sample['img'])
        label = np.array(sample['mask'])
        offset = np.random.randint(-self.max_offset, self.max_offset)
        h, w = img.shape[:2]

        if offset > 0:
            img[offset:] = img[:h - offset]
            img[:offset] = 0
            label[offset:] = label[:h - offset]
            label[:offset] = 0
        elif offset < 0:
            real_offset = -offset
            img[:h - real_offset] = img[real_offset:]
            img[h - real_offset:] = 0
            label[:h - real_offset] = label[real_offset:]
            label[h - real_offset:] = 0

        sample['img'] = img
        sample['mask'] = label
        return sample


class Resize:
    """Resize the image and optionally the mask to a given size."""

    def __init__(self, size, cfg=None):
        assert isinstance(size, collections.abc.Iterable) and len(size) == 2, \
            "Size must be a tuple of (height, width)."
        self.size = size

    def __call__(self, sample):
        sample['img'] = cv2.resize(sample['img'], self.size, interpolation=cv2.INTER_CUBIC)
        if 'mask' in sample:
            sample['mask'] = cv2.resize(sample['mask'], self.size, interpolation=cv2.INTER_NEAREST)
        return sample


class RandomCrop:
    """Randomly crop the image to the specified size."""

    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[:2]
        th, tw = self.size

        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        return [img[h1:h2, w1:w2] for img in img_group]


class CenterCrop:
    """Center crop the image to the specified size."""

    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[:2]
        th, tw = self.size

        h1 = max(0, int((h - th) / 2))
        w1 = max(0, int((w - tw) / 2))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        return [img[h1:h2, w1:w2] for img in img_group]


class RandomRotation:
    """Randomly rotate the image and mask within a specified degree range."""

    def __init__(self, degree=(-10, 10), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=None, cfg=None):
        self.degree = degree
        self.interpolation = interpolation
        self.padding = padding if padding is not None else [0, 0]

    def _rotate_img(self, sample, map_matrix):
        h, w = sample['img'].shape[:2]
        sample['img'] = cv2.warpAffine(sample['img'], map_matrix, (w, h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=self.padding)

    def _rotate_mask(self, sample, map_matrix):
        if 'mask' in sample:
            h, w = sample['mask'].shape[:2]
            sample['mask'] = cv2.warpAffine(sample['mask'], map_matrix, (w, h),
                                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=self.padding)

    def __call__(self, sample):
        if random.random() < 0.5:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = sample['img'].shape[:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
            self._rotate_img(sample, map_matrix)
            self._rotate_mask(sample, map_matrix)
        return sample


class RandomBlur:
    """Randomly apply Gaussian blur to the image."""

    def __init__(self, applied, cfg=None):
        self.applied = applied

    def __call__(self, img_group):
        if random.random() < 0.5:
            return [
                cv2.GaussianBlur(img, (5, 5), random.uniform(1e-6, 0.6)) if a else img
                for img, a in zip(img_group, self.applied)
            ]
        return img_group


class RandomHorizontalFlip:
    """Randomly horizontally flip the image and mask with a probability of 0.5."""

    def __init__(self, cfg=None):
        pass

    def __call__(self, sample):
        if random.random() < 0.5:
            sample['img'] = np.fliplr(sample['img'])
            if 'mask' in sample:
                sample['mask'] = np.fliplr(sample['mask'])
        return sample


class Normalize:
    """Normalize the image by subtracting the mean and dividing by the standard deviation."""

    def __init__(self, img_norm, cfg=None):
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)

    def __call__(self, sample):
        img = sample['img']
        if len(self.mean) == 1:
            img = (img - self.mean) / self.std
        else:
            img = (img - self.mean[np.newaxis, np.newaxis, ...]) / self.std[np.newaxis, np.newaxis, ...]
        sample['img'] = img
        return sample


def CLRTransforms(img_h, img_w):
    """Define a series of transformations to apply to the images."""
    return [
        dict(name='Resize', parameters=dict(size=(img_h, img_w))),
        dict(name='RandomCrop', parameters=dict(size=(img_h, img_w))),
        dict(name='RandomHorizontalFlip'),
        dict(name='ToTensor'),
        dict(name='Normalize', parameters=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),
    ]


def process(sample, transforms):
    """Apply the list of transformations to the sample."""
    for transform in transforms:
        tname = transform['name']
        tparams = transform['parameters'] if 'parameters' in transform else {}
        sample = globals()[tname](**tparams)(sample)
    return sample

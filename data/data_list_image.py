#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class ImageListIndex(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


class ImageValueList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=rgb_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel - mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
    """

    def __init__(self, mean=None, meanfile=None):
        if mean:
            self.mean = mean
        else:
            arr = np.load(meanfile)
            self.mean = torch.from_numpy(arr.astype('float32')/255.0)[[2,1,0],:,:]

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m in zip(tensor, self.mean):
            t.sub_(m)
        return tensor

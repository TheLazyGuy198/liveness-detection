from torchvision import transforms
import torch
import logging
import numpy as np
import random
from PIL import Image
import albumentations as A


class ShufflePatch(torch.nn.Module):

    def __init__(self, patch_size=32, shuffle_prob=0.5, config=None):
        super(ShufflePatch, self).__init__()
        self.patch_size = patch_size
        self.shuffle_prob = shuffle_prob

    def shuffle_patches(self, im):
        img_patches = []
        H, W, C = im.shape

        num_patch_h = H // self.patch_size
        num_patch_w = W // self.patch_size

        indices_w = np.linspace(0, W, num_patch_w + 1, endpoint=True, dtype=int)
        indices_h = np.linspace(0, H, num_patch_h + 1, endpoint=True, dtype=int)

        patches = []
        for i in range(num_patch_w):
            for j in range(num_patch_h):
                start_w, end_w = indices_w[i], indices_w[i + 1]
                start_h, end_h = indices_h[i], indices_h[i + 1]
                patch = im[start_h:end_h, start_w:end_w, :]
                patches.append(patch)
        random.shuffle(patches)
        new_im = np.zeros_like(im)
        for i in range(num_patch_w):
            for j in range(num_patch_h):
                start_w, end_w = indices_w[i], indices_w[i + 1]
                start_h, end_h = indices_h[i], indices_h[i + 1]
                new_im[start_h:end_h, start_w:end_w, :] = patches[i*num_patch_w+j]

        return new_im

    def forward(self, im):
        p = np.random.uniform()
        if p < self.shuffle_prob:
            if 'PIL' in str(type(im)):
                # PIL to im narray
                im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
                im = self.shuffle_patches(im)
                return Image.fromarray(im.astype(np.uint8))
        return im


def get_augmentation_transforms(config):
    augmentation_transforms = []
    if config.TRAIN.AUG.ColorJitter.ENABLE:
        logging.info('Data Augmentation ColorJitter is ENABLED')
        brightness = config.TRAIN.AUG.ColorJitter.brightness
        contrast = config.TRAIN.AUG.ColorJitter.contrast
        saturation = config.TRAIN.AUG.ColorJitter.saturation
        hue = config.TRAIN.AUG.ColorJitter.hue
        augmentation_transforms.append(transforms.ColorJitter(brightness, contrast, saturation, hue))

    if config.TRAIN.AUG.RandomHorizontalFlip.ENABLE:
        logging.info('Data Augmentation RandomHorizontalFlip is ENABLED')
        p = config.TRAIN.AUG.RandomHorizontalFlip.p
        augmentation_transforms.append(transforms.RandomHorizontalFlip(p))
        augmentation_transforms.append(A.HorizontalFlip(p))

    if config.TRAIN.AUG.RandomCrop.ENABLE:
        logging.info('Data Augmentation RandomCrop is ENABLED')
        size = config.TRAIN.AUG.RandomCrop.size
        augmentation_transforms.append(transforms.RandomCrop(size))
        augmentation_transforms.append(A.RandomCrop(height=size, width=size))

    if config.TRAIN.AUG.RandomErasing.ENABLE:
        logging.info('Data Augmentation RandomErasing is ENABLED')
        p = config.TRAIN.AUG.RandomErasing.p
        scale = config.TRAIN.AUG.RandomErasing.scale
        ratio = config.TRAIN.AUG.RandomErasing.ratio
        augmentation_transforms.append(A.RandomE)
        augmentation_transforms.extend([transforms.ToTensor(),
                                        transforms.RandomErasing(p=p, scale=scale, ratio=ratio),
                                        transforms.ToPILImage()])

    if config.TRAIN.AUG.ShufflePatch.ENABLE:
        logging.info('Data Augmentation RandomErasing is ENABLED')
        p = config.TRAIN.AUG.ShufflePatch.p
        size = config.TRAIN.AUG.ShufflePatch.size
        shuffle_patch_transform = ShufflePatch(patch_size=size, shuffle_prob=p)
        augmentation_transforms.extend([shuffle_patch_transform])

    return augmentation_transforms


class VisualTransform(torch.nn.Module):
    def __init__(self, config, augmentation_transforms=[]):
        super(VisualTransform, self).__init__()
        self.config = config
        img_size = config.DATA.IN_SIZE
        self.transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size))
        ]
        self.transform_list.extend(augmentation_transforms)
        self.transform_list.append(transforms.ToTensor())

        if self.config.DATA.NORMALIZE.ENABLE:
            norm_transform = transforms.Normalize(mean=self.config.DATA.NORMALIZE.MEAN,
                                                  std=self.config.DATA.NORMALIZE.STD)
            self.transform_list.append(norm_transform)
            logging.info("Mean STD Normalize is ENABLED")

        self.transforms = transforms.Compose(self.transform_list)

    def forward(self, x):
        return self.transforms(x)

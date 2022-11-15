import cv2
from torchvision import datasets
from torch.utils.data import DataLoader

from utils.misc import get_transforms, onehot


def opencv_loader(path):
    img = cv2.imread(path)
    return img


class FrameRecapDataset(datasets.ImageFolder):

    def __init__(self, root, mode='train', use_transform=False, img_width=240, img_height=240):
        super(FrameRecapDataset, self).__init__(root, loader=opencv_loader)
        self.img_w = img_width
        self.img_h = img_height
        self.use_transform = use_transform
        self.mode = mode

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.use_transform:
            transform = get_transforms(self.mode, target, self.img_w, self.img_h)
        else:
            transform = None

        if transform is not None:
            try:
                sample = transform(image=sample)['image']
            except Exception as err:
                print('Error Occured: %s' % err, path)

        target = onehot(2, target)
        return sample, target


def get_train_loader(cfg):
    root_path = '{}/train'.format(cfg.train_root_path)
    train_ds = FrameRecapDataset(root_path, mode='train',
                                 use_transform=True, img_width=cfg.width, img_height=cfg.height)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    return train_loader


def get_val_loader(cfg):
    root_path = '{}/val'.format(cfg.train_root_path)
    val_ds = FrameRecapDataset(root_path, mode='val', use_transform=True,
                               img_width=cfg.width, img_height=cfg.height)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    return val_loader

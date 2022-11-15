import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import random
import os

from utils.misc import get_transforms, onehot


class VideoRecapDataset(Dataset):

    def __init__(self, root, mode='train', use_transform=False, img_width=240, img_height=240):
        super(VideoRecapDataset, self).__init__()
        self.paths = glob(root + '/**/*.mp4', recursive=True)
        self.img_w = img_width
        self.img_h = img_height
        self.use_transform = use_transform
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        d = os.path.dirname(self.paths[index])
        target = int(d.split('/')[-1])
        vid_cap = cv2.VideoCapture(self.paths[index])
        total_frames = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        succ, frame = False, None
        while not succ:
            rand_frame_id = random.randint(0, total_frames)
            vid_cap.set(cv2.CAP_PROP_POS_FRAMES, rand_frame_id)
            succ, frame = vid_cap.read()

        if self.use_transform:
            transform = get_transforms(self.mode, target, self.img_w, self.img_h)
        else:
            transform = None

        if transform is not None:
            try:
                sample = transform(image=frame)['image']
            except Exception as err:
                print('Error Occured: %s' % err, self.paths[index])

        target = onehot(2, target)

        return sample, target


def get_train_loader(cfg):
    root_path = '{}/train'.format(cfg.train_root_path)
    train_ds = VideoRecapDataset(root_path, mode='train',
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
    val_ds = VideoRecapDataset(root_path, mode='val', use_transform=True,
                               img_width=cfg.width, img_height=cfg.height)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )

    return val_loader

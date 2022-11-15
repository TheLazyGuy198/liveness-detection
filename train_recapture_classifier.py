import torch
import os
from easydict import EasyDict
import argparse

from models.recapture_classifier import RecaptureClassifier
# from utils.datasets.video_recapture_dataset import get_train_loader, get_val_loader
from utils.datasets.frame_recapture_dataset import get_train_loader, get_val_loader
from trainer.recap_trainer import RecapTrainer
from utils.misc import get_state_dict


def train_config():
    cfg = EasyDict()
    cfg.lr = 1e-3
    cfg.step_scheduler = True
    cfg.batch_size = 16
    cfg.epochs = 100
    cfg.arch = 'tf_efficientnetv2_b1'
    cfg.device_id = 0
    cfg.train_root_path = '/mnt/data/minhnq54/passport-train/cropped-passport-images/'
    cfg.pretrained_path = None
    cfg.width = 320
    cfg.height = 320
    cfg.wts_path = './out'
    cfg.log_path = './logs'
    cfg.verbose = True
    cfg.save_every = 40
    return cfg


def update_config(args, cfg):
    cfg.device = args.device_id
    cfg.arch = args.model
    cfg.train_root_path = cfg.train_root_path if args.train_path is None else args.train_path
    cfg.batch_size = cfg.batch_size if args.batch_size is None else args.batch_size
    cfg.epochs = cfg.epochs if args.epochs is None else args.epochs
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", type=int, default=0, help="which gpu id, 0123")
    parser.add_argument("--train-path", type=str, default=None, help='Path to training dataset')
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train model')
    args = parser.parse_args()

    cfg = train_config()
    cfg = update_config(args, cfg)

    model = RecaptureClassifier(arch=cfg.arch)
    device = "cuda:{}".format(cfg.device_id) if torch.cuda.is_available() else "cpu"

    if cfg.pretrained_path is not None:
        state_dict = torch.load(cfg.pretrained_path, map_location=device)
        model.load_state_dict(state_dict)

    train_loader = get_train_loader(cfg)
    val_loader = get_val_loader(cfg)
    trainer = RecapTrainer(model, device, cfg)
    best_model = trainer.train(train_loader, val_loader)
    torch.save(get_state_dict(best_model),
               os.path.join('weights/recapture/', cfg.arch + '-' + str(cfg.width) + '_' + str(cfg.height) + '.pth'))

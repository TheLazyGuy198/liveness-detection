import os
import torch
import copy
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from pathlib import Path
from datetime import datetime

from utils.metrics import LabelSmoothing, Average, RocAuc, APScore, F1Score, Accuracy
from utils.misc import get_state_dict


class RecapTrainer:

    def __init__(self, model, device, cfg):
        self.cfg = cfg
        Path(cfg.wts_path).mkdir(parents=True, exist_ok=True)
        (Path(cfg.log_path) / 'scalar' / self.cfg.arch / datetime.now().strftime('%Y%m%d-%H%M')).mkdir(parents=True,
                                                                                                       exist_ok=True)
        (Path(cfg.log_path) / 'history' / self.cfg.arch).mkdir(parents=True, exist_ok=True)
        self.wts_path = cfg.wts_path

        self.scalar_path = os.path.join(cfg.log_path, 'scalar', self.cfg.arch, datetime.now().strftime('%Y%m%d-%H%M'))
        self.log_path = os.path.join(cfg.log_path, 'history', self.cfg.arch, datetime.now().strftime('%Y%m%d-%H%M') + '.txt')
        self.model = model
        self.device = device

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=16,
            verbose=False,
            threshold=0.0001,
            threshold_mode='abs',
            cooldown=0,
            min_lr=1e-8,
            eps=1e-08
        )
        self.criterion = LabelSmoothing().to(self.device)

        self.logger = SummaryWriter(self.scalar_path)
        self.best_val_acc = 0.
        self.best_f1_score = 0.
        self.best_model = None

    def train_per_epoch(self, train_loader):
        self.model.train()
        avg_loss = Average()
        roc_auc = RocAuc()
        ap_score = APScore()
        f1_score = F1Score()
        acc = Accuracy()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Training')

        for step, (images, targets) in pbar:
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()

            avg_loss.update(loss.detach().item(), batch_size)
            roc_auc.update(targets, outputs)
            ap_score.update(targets, outputs)
            f1_score.update(targets, outputs)
            acc.update(targets, outputs)

            self.optimizer.step()
            if self.cfg.step_scheduler:
                self.scheduler.step(metrics=avg_loss.avg)

        return avg_loss, roc_auc, ap_score, f1_score, acc

    def val_per_epoch(self, val_loader):
        self.model.eval()
        avg_loss = Average()
        roc_auc = RocAuc()
        ap_score = APScore()
        f1_score = F1Score()
        acc = Accuracy()

        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Validating')

        for step, (images, targets) in pbar:
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                avg_loss.update(loss.detach().item(), batch_size)
                roc_auc.update(targets, outputs)
                ap_score.update(targets, outputs)
                f1_score.update(targets, outputs)
                acc.update(targets, outputs)

        return avg_loss, roc_auc, ap_score, f1_score, acc

    def train(self, train_loader, val_loader):
        self.model.to(self.device)
        self.best_model = copy.deepcopy(self.model)

        for e in range(self.cfg.epochs):
            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.utcnow().isoformat()
            self.log(f'\n{timestamp}\nLR: {lr}')

            avg_loss, roc_auc, ap_score, f1_score, acc = self.train_per_epoch(train_loader)
            self.logger.add_scalar('Training/Loss', avg_loss.avg, e)
            self.logger.add_scalar('Training/ROC-AUC', roc_auc.avg, e)
            self.logger.add_scalar('Training/F1 Score', f1_score.avg, e)
            self.logger.add_scalar('Training/Accuracy', acc.avg, e)
            self.logger.add_scalar('Training/AP', ap_score.avg, e)
            self.log(
                msg=f'Epoch {e + 1}/{self.cfg.epochs}, train_avg_loss: {avg_loss.avg:.5f}, '
                    f'train_roc_auc: {roc_auc.avg:.5f}, train_ap: {ap_score.avg:.5f}, '
                    f'train_f1_score: {f1_score.avg:.5f}, train_acc: {acc.avg}'
            )

            avg_loss, roc_auc, ap_score, f1_score, acc = self.val_per_epoch(val_loader)
            self.logger.add_scalar('Validation/Loss', avg_loss.avg, e)
            self.logger.add_scalar('Validation/ROC-AUC', roc_auc.avg, e)
            self.logger.add_scalar('Validation/F1 Score', f1_score.avg, e)
            self.logger.add_scalar('Validation/Accuracy', acc.avg, e)
            self.logger.add_scalar('Validation/AP', ap_score.avg, e)
            self.log(
                msg=f'Epoch {e + 1}/{self.cfg.epochs}, val_avg_loss: {avg_loss.avg:.5f}, '
                    f'val_roc_auc: {roc_auc.avg:.5f}, val_ap: {ap_score.avg:.5f}, '
                    f'val_f1_score: {f1_score.avg:.5f}, val_acc: {acc.avg}'
            )

            if f1_score.avg > self.best_f1_score:
                print(f'F1-score improved from {self.best_f1_score:.5f} to {f1_score.avg:.5f}')
                self.best_f1_score = f1_score.avg
                self.model.eval()
                self.save_best(os.path.join(self.wts_path, 'best_' + self.cfg.arch + '_' + str(self.cfg.height)
                                            + '_' + str(self.cfg.width) + '.bin'))
                self.best_model = copy.deepcopy(self.model)

            if (e + 1) % self.cfg.save_every == 0:
                self.model.eval()
                self.save_snapshot(os.path.join(self.wts_path, self.cfg.arch + 'epoch_' + str(e + 1) + '_' + str(self.cfg.height)
                                            + '_' + str(self.cfg.width) + '.bin'))

        return self.best_model

    def log(self, msg):
        if self.cfg.verbose:
            print(f'{msg}\n')
        with open(self.log_path, 'a+') as fout:
            fout.write(f'{msg}\n')

    def save_best(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': get_state_dict(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_val_acc
        }, path)

    def save_snapshot(self, path):
        self.model.eval()
        torch.save(get_state_dict(self.model), path)

import numpy as np
import torch
from torchvision.utils import make_grid

import utils
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils import global_var, util

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)  # 返回可迭代对象的迭代长度
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(len(data_loader)/6)  # 用来控制多少步显示一次输出int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()  # 调用的是nn.model.train()设置为训练模式
        self.train_metrics.reset()  # 运行本次epoch之前，先把跟踪日志清0重置
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # timer = utils.MyTimer(f'epoch {epoch} have total {len(self.data_loader)} batch. Batch {batch_idx + 1} GPU cal run')

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # timer.stop()

            # timer = utils.MyTimer(f'epoch {epoch} have total {len(self.data_loader)} batch. Batch {batch_idx + 1} CPU IO run')

            # 传入当前运行属于第几个step（第几个batch）
            # 当前的epoch次序*单个epoch的总step数+当前epoch的step次序
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss.item():.6f}')
                global_var.get_value('email_log').add_log(f'Train Epoch: {epoch} {self._progress(batch_idx)} Loss: {loss.item():.6f}')
                # self.writer.add_image('input', make_grid(data, nrow=8))  # 一个巨耗时间，普通训练一个batch能0.05s，这能飙到几十秒

            if batch_idx == self.len_epoch:
                break

            # timer.stop()
        log = self.train_metrics.result()

        # timer = utils.MyTimer(f'epoch val run')

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        # timer.stop()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))  # 耗时

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        # 输出str，用于生成训练进度的str的函数
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

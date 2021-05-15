import argparse
from typing import Any, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import Metric, AveragePrecision


# Define default CMD arguments
OPTIMIZER = 'Adam'
LR = 1e-3
LOSS = 'cross_entropy'
ONE_CYCLE_TOTAL_STEPS = 100
N_CLASSES = 5


class MeanAveragePrecision(AveragePrecision):
    def __init__(self, num_classes: Optional[int] = None,
            pos_label: Optional[int] = None,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group = None) -> None:
        super().__init__(
            num_classes=num_classes,
            pos_label=pos_label,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group
        )

    def compute(self) -> torch.Tensor:
        avg_precision = super().compute()
        return torch.Tensor(np.nanmean(avg_precision))


class BaseLitModel(pl.LightningModule):  # pylint: disable=too-many-ancestors
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get('lr', LR)

        loss = self.args.get('loss', LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        self.one_cycle_max_lr = self.args.get('one_cycle_max_lr', None)
        self.one_cycle_total_steps = self.args.get('one_cycle_total_steps', ONE_CYCLE_TOTAL_STEPS)

        num_classes = self.args.get('n_classes', N_CLASSES)
        self.train_map = MeanAveragePrecision(num_classes)
        self.val_map = MeanAveragePrecision(num_classes)
        self.test_map = MeanAveragePrecision(num_classes)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='optimizer class from torch.optim')
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argument('--one_cycle_max_lr', type=float, default=None)
        parser.add_argument('--one_cycle_total_steps', type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument('--loss', type=str, default=LOSS, help='loss function from torch.nn.functional')
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        self.log('train_loss', loss)
        self.train_map(probs, y)
        self.log('train_mAP', self.train_map, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probs = torch.softmax(logits, dim=1)
        self.log('val_loss', loss, prog_bar=True)
        self.val_map(probs, y)
        self.log('val_mAP', self.val_map, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        self.test_map(probs, y)
        self.log('test_mAP', self.test_map, on_step=False, on_epoch=True)

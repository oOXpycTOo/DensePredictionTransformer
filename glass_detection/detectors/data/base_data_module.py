"""Base data module class with utilities"""
import argparse
from pathlib import Path
from typing import Tuple, Union

import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader

from .base_dataset import BaseDataset


BATCH_SIZE=128
NUM_WORKERS=0


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get('gpus'), (str, int))

        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.data_train: Union[BaseDataset, ConcatDataset]
        self.data_val: Union[BaseDataset, ConcatDataset]
        self.data_test: Union[BaseDataset, ConcatDataset]

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[3] / 'data'

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size", type=int, default=BATCH_SIZE, help="Number of examples to operate on per forward step."
        )
        parser.add_argument(
            "--num_workers", type=int, default=NUM_WORKERS, help="Number of additional processes to load data."
        )
        return parser

    def config(self):
        return {'input_dims': self.dims, 'output_dims': self.output_dims}

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, *args, **kwargs) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu
        )
    
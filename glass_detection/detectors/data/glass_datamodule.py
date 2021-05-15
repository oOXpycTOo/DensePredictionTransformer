import argparse
import json
from typing import Any, Dict
from pathlib import Path

from .base_data_module import BaseDataModule
from .glass_dataset import GlassSegmentationDataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np


IMG_HEIGHT = 384
IMG_WIDTH = 384
PATCH_SIZE = 16
N_CLASSES = 5
DATA_FOLDER = BaseDataModule.data_dirname()


def parse_metadata(path: Path) -> Dict[str, Any]:
    with open(path) as file:
        return json.load(file)


class GlassSegmentationDataModule(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_folder = Path(self.args.get('data_folder', DATA_FOLDER))
        self.height = self.args.get('image_height', IMG_HEIGHT)
        self.width = self.args.get('image_width', IMG_WIDTH)
        self.patch_size = self.args.get('patch_size', PATCH_SIZE)
        self.n_classes = self.args.get('n_classes', N_CLASSES)
        self.dims = (3, self.height, self.width)
        self.output_dims = (self.n_classes, self.height, self.width)
        self.metadata = parse_metadata(self.data_folder / 'metainfo.json')

        self.train_val_transform = A.Compose([
            A.Resize(self.height, self.width),
            A.HorizontalFlip(p=0.5),
            A.ISONoise(),
            A.ColorJitter(),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ]
        )
        self.test_transform = A.Compose([
            A.Resize(self.height, self.width),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ]
        )

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument('--image_height', type=int, default=IMG_HEIGHT,
            help='Image height to be used for training and validation')
        parser.add_argument('--image_width', type=int, default=IMG_WIDTH,
            help='Image width to be used for training and validation')
        parser.add_argument('--patch_size', type=int, default=PATCH_SIZE,
            help='Patch size to be used for image split')
        parser.add_argument('--n_classes', type=int, default=N_CLASSES,
                            help='Number of segmentation classes')
        parser.add_argument('--data_folder', type=str, default=DATA_FOLDER,
            help='Path to a folder with image data')

        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def config(self) -> Dict[str, Any]:
        return {'input_dims': self.dims, 'output_dims': self.output_dims,
                'height': self.height, 'width': self.width, 'patch_size': self.patch_size,
                'n_classes': self.n_classes}

    def setup(self, stage: str = None) -> None:
        palette = np.array(self.metadata['palette'])
        classes = self.metadata['classes']
        if stage == 'fit' or stage is None:
            train_files = self.metadata['train']
            val_files = self.metadata['val']
            self.data_train = GlassSegmentationDataset(self.data_folder, train_files,
                                                       classes, palette, transform=self.train_val_transform)
            self.data_val = GlassSegmentationDataset(self.data_folder, val_files,
                                                     classes, palette, transform=self.train_val_transform)

        if stage == 'test' or stage is None:
            test_files = self.metadata['test']
            self.data_test = GlassSegmentationDataset(self.data_folder, test_files,
                                                      classes, palette, transform=self.test_transform)
        
    def __repr__(self) -> str:
        basic = ('GlassSegmentation dataset\n',
                f'Image height: {self.height}\n',
                f'Image width: {self.width}\n',
                f'Patch size: {self.patch_size}'
                f'Data folder: {self.data_folder}\n')
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic
        return basic

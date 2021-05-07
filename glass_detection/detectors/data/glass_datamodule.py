from glass_detection.detectors.data.base_dataset import BaseDataset
from glass_detection.detectors.data.base_data_module import BaseDataModule

import argparse

import albumentations as A

NUM_TRAIN = 100
NUM_VAL = 10
NUM_TEST = 10
IMG_HEIGHT = 256
IMG_WIDTH = 256
DATA_FOLDER = BaseDataModule.data_dirname


class GlassSegmentationDataModule(BaseDataModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        self.num_train = self.args.get('num_train', NUM_TRAIN)
        self.num_val = self.args.get('num_val', NUM_VAL)
        self.num_test = self.args.get('num_test', NUM_TEST)
        self.height = self.args.get('image_height', IMG_HEIGHT)
        self.width = self.args.get('image_width', IMG_WIDTH)
        self.data_folder = self.args.get('data_folder', DATA_FOLDER)

        self.dims = (3, self.height, self.width)
        self.output_dims = (1, self.height, self.width)
        self.train_val_transform = A.Compose(
            A.Resize(self.height, self.width)
            A.HorizontalFlip(p=0.5),
            A.ISONoise(),
            A.ColorJitter()
        )
        self.test_transform = A.Compose(
            A.Resize(self.height, self.width)
        )

    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser) -> None:
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument('--image_height', type=int, default=IMG_HEIGHT,
            help='Image height to be used for training and validation')
        parser.add_argument('--image_width', type=int, default=IMG_WIDTH,
            help='Image width to be used for training and validation')
        parser.add_argument('--data_folder', type=str, default=DATA_FOLDER,
            help='Path to a folder with image data')

        return parser

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            train_folder = self.data_folder / 'train'
            val_folder = self.data_folder / 'val'
            self.data_train = GlassDataset(train_folder, transform=self.train_val_transform)
            self.data_val = GlassDataset(val_folder, transform=self.train_val_transform)

        if stage == 'test' or stage is None:
            test_folder = self.data_folder / 'test'
            self.data_test = GlassDataset(test_folder)
        
    def __repr__(self) -> str:
        basic = ('GlassSegmentation dataset\n',
                f'Image height: {self.image_height}\n',
                f'Image width: {self.image_width}\n',
                f'Data folder: {self.data_folder}\n')
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic
        return basic

from argparse import Namespace
import json
import os
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

DEFAULT_DIR = Path(__file__).resolve().parents[3] / 'data'


def convert_colors_to_labels(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    difference = np.sum(np.abs(mask[:, :, None] - palette[None, None]), axis=3)
    labels = np.argmin(difference, axis=2)
    return labels


def convert_labels_to_colors(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[labels]


class ImageCallback(pl.Callback):
    def __init__(self, args: Namespace, num_read: int = 3) -> None:
        super(ImageCallback, self).__init__()
        args = vars(args)
        self.height = args.get('height')
        self.width = args.get('width')
        self.data_folder = Path(args.get('data_folder', DEFAULT_DIR))
        with open(self.data_folder / 'metainfo.json') as file:
            metainfo = json.load(file)
        self.palette = np.array(metainfo['palette'])
        self.num_read = num_read
        self.images, self.masks = self.read_files_into_images(metainfo['val'])
        self.transform = A.Compose([
            A.Resize(self.height, self.width),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ]
        )

    def read_files_into_images(self, files: List[str]) -> Tuple[np.array, np.array]:
        images = []
        masks = []
        selected = np.random.choice(files, self.num_read, replace=False)
        for filename in selected:
            image = cv2.imread(str(self.data_folder / 'images' / f'{filename}.jpg'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_path = str(self.data_folder / 'masks' / f'{filename}.png')
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path)
            else:
                mask = np.zeros_like(image)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            images.append(image)
            masks.append(mask)
        return images, masks

    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        examples = []
        with torch.no_grad():
            for i in range(self.num_read):
                transformed = self.transform(image=self.images[i], mask=self.masks[i])
                image = transformed['image'].unsqueeze(0).to(pl_module.device)
                mask = transformed['mask'].cpu().numpy()
                preds = pl_module(image)
                pred_label = torch.argmax(preds, 1)[0].detach().cpu().numpy()
                pred_color = convert_labels_to_colors(pred_label, self.palette)
                image = (image * 0.5) + 0.5
                image = image[0].permute(1, 2, 0).detach().cpu().numpy()
                result = np.hstack([image, mask / 255., pred_color / 255.])
                examples.append(wandb.Image(result, caption='Image, True, Predicted'))
            trainer.logger.experiment.log({'val_results': examples})

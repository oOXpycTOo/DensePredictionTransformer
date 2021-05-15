import os
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch

from .base_dataset import BaseDataset


class GlassSegmentationDataset(BaseDataset):
    def __init__(self, data_folder: Path,
                 files: List[str],
                 classes: List[str],
                 palette: np.array,
                 transform: Callable) -> None:
        super().__init__(transform)
        self.classes = classes
        self.palette = palette
        self.image_folder = data_folder / 'images'
        self.mask_folder = data_folder / 'masks'
        self.data = files
        masks_files = set(os.listdir(self.mask_folder))
        self.targets = [file if file in masks_files else '' for file in files]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_filename = self.data[idx]
        mask_filename = self.targets[idx]
        image = cv2.imread(str(self.image_folder / image_filename))
        if mask_filename:
            mask = cv2.imread(str(self.mask_folder / mask_filename))
        else:
            mask = np.zeros_like(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = convert_colors_to_labels(mask, self.palette)
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask'].long()


def convert_colors_to_labels(mask: np.array, palette: np.array) -> np.array:
    difference = np.sum(np.abs(mask[:, :, None] - palette[None, None]), axis=3)
    labels = np.argmin(difference, axis=2)
    return labels

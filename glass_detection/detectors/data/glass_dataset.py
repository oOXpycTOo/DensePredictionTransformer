import json
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch

from glass_detection.detectors.data.base_dataset import BaseDataset


class GlassSegmentationDataset(BaseDataset):
    def __init__(self, data_folder: Path, transform: Callable, target_transform: Callable) -> None:
        super().__init__(transform, target_transform)
        parent_folder = data_folder.parents[0]
        self.classes, self.palette = self.__parse_metadata(parent_folder)
        self.image_folder = self.data_folder / 'images'
        self.mask_folder = self.data_folder / 'masks'
        self.data = self.__read_files_from_folder(self.image_folder)
        self.targets = self.__read_files_from_folder(self.mask_folder)

    def __parse_metadata(self, folder: Path) -> Tuple[List[str], np.ndarray]:
        with open(folder / 'metadata.json') as metafile:
            metadata = json.load(metafile)
        classes = []
        palette = []
        for classname, color in metadata.items():
            classes.append(classname)
            palette.append(color[])
        return classes, np.array(palette)[:, ::-1]

    def __read_files_from_folder(self, folder: Path) -> List[str]:
        filenames = []
        for filename in self.image_folder.iterdir():
            if filename.is_file():
                filenames.append(str(filename))
        return filenames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_filename = self.data[idx]
        mask_filename = self.targets[idx]
        image = cv2.imread(image_filename)
        mask = cv2.imread(mask_filename)
        mask = convert_colors_to_labels(mask, self.palette)
        return self.transfrom(image), self.target_transform(mask)

def convert_colors_to_labels(mask: np.array, palette: np.array) -> np.array:
    mask [h, w, 3]
    palette [n_cl, 3]
    difference = np.sum(np.abs(mask[:, :, None] - palette[None, None]), dim=3)
    labels = np.argmin(difference, dim=2)
    return labels

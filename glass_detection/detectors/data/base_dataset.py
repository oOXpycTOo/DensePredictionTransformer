from torch.utils.data import Dataset
from typing import Any, Callable, Tuple


class BaseDataset(Dataset):
    def __init__(self,
                data,
                targets,
                transform: Callable = None,
                target_transform: Callable = None) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        datum, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            datum = self.transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return datum, target

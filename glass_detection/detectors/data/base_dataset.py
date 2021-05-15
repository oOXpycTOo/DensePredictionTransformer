from torch.utils.data import Dataset
from typing import Any, Callable


class BaseDataset(Dataset):
    def __init__(self,
                transform: Callable = None,
                target_transform: Callable = None) -> None:
        super().__init__()
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        pass

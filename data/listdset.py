from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, list_file_path: str, transform: Optional[Callable] = None) -> None:
        super().__init__()

        self.items: List[Tuple[Image.Image, Tensor]] = []

        with open(list_file_path) as f:
            lines = [line.strip() for line in f.readlines()]

        for line in lines:
            img_path, label = line.split(",")
            img = Image.open(img_path).convert("RGB")
            label = torch.tensor(int(label), dtype=torch.long)
            self.items.append((img, label))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[Any, Tensor]:
        img, label = self.items[index]
        if self.transform:
            img = self.transform(img)

        return img, label

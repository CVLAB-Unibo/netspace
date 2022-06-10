from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from trainers.utils import progress_bar


class SDFDataset(Dataset):
    def __init__(self, root: Path, num_samples: int, indices: List[int] = []) -> None:
        super().__init__()

        self.num_samples = num_samples

        self.npz_files: List[Path] = list(root.glob("*.npz"))
        if len(indices) > 0:
            self.npz_files = [self.npz_files[i] for i in indices]

        self.sdfs: List[Tuple[Tensor, Tensor]] = []
        for npz_file in progress_bar(self.npz_files, desc="Loading SDF dataset"):
            data = np.load(npz_file)

            pos = torch.from_numpy(data["pos"])
            pos_nans = torch.isnan(pos[:, 3])
            pos = pos[~pos_nans, :]

            neg = torch.from_numpy(data["neg"])
            neg_nans = torch.isnan(neg[:, 3])
            neg = neg[~neg_nans, :]

            self.sdfs.append((pos, neg))

    def __len__(self) -> int:
        return len(self.npz_files)

    def __getitem__(self, index: int) -> Tensor:
        pos, neg = self.sdfs[index]

        num = int(self.num_samples / 2)
        random_pos = (torch.rand(num) * pos.shape[0]).long()
        samples_pos = torch.index_select(pos, 0, random_pos)

        random_neg = (torch.rand(num) * neg.shape[0]).long()
        samples_neg = torch.index_select(neg, 0, random_neg)

        samples = torch.cat([samples_pos, samples_neg], 0)
        return samples

    def get_sdf(self, index: int) -> Tuple[str, Tensor]:
        pos, neg = self.sdfs[index]
        sdf = torch.cat([pos, neg], 0)
        return self.npz_files[index].stem, sdf

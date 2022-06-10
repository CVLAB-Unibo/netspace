from pathlib import Path
from typing import Tuple

import torch
from models.mlp import MLP
from torch import Tensor
from torch.utils.data import Dataset


class MLPDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        range: Tuple[int, int],
        prep_shape: Tuple[int, int],
        num_coords: int,
    ) -> None:
        super().__init__()

        ckpt_paths = sorted(list(dataset_root.glob("*.pt")))
        self.ckpt_paths = ckpt_paths[range[0] : range[1]]

        self.prep_shape = prep_shape
        self.num_coords = num_coords

    def __len__(self) -> int:
        return len(self.ckpt_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        ckpt_path = self.ckpt_paths[index]
        ckpt = torch.load(ckpt_path)

        state_dict = ckpt["mlp"]
        params = list(state_dict.values())
        prep = MLP.get_prep(params, self.prep_shape)
        prep = prep.detach()

        sdf = ckpt["sdf"]
        half = sdf.shape[0] // 2
        pos, neg = sdf[:half, :3], sdf[half:, :3]

        num = int(self.num_coords / 2)
        random_pos = (torch.rand(num) * pos.shape[0]).long()
        coords_pos = torch.index_select(pos, 0, random_pos)

        random_neg = (torch.rand(num) * neg.shape[0]).long()
        coords_neg = torch.index_select(neg, 0, random_neg)

        coords = torch.cat([coords_pos, coords_neg], 0)

        return prep, coords

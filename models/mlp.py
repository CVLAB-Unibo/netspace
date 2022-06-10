from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)


def weights_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_weights_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, num_hidden_layers: int) -> None:
        super().__init__()

        layers = []

        layers.append(nn.Linear(3, hidden_dim))
        layers.append(Sine())
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Sine())
        layers.append(nn.Linear(hidden_dim, 1))

        self.mlp = nn.Sequential(*layers)

        self.mlp.apply(weights_init)
        self.mlp[0].apply(first_layer_weights_init)

    def forward(self, coordinates: Tensor) -> Tensor:
        return self.mlp(coordinates).squeeze()

    def num_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    @staticmethod
    def get_prep(params: List[Tensor], prep_shape: Tuple[int, int]) -> Tensor:
        height = prep_shape[0]
        width = prep_shape[1]

        all_params = torch.cat([p.view(-1) for p in params], dim=0)

        rows = []
        start = 0
        while start < len(all_params):
            row = all_params[start : start + width]
            row = F.pad(row, [0, width - row.numel()], "constant", 0)
            rows.append(row)
            start += width

        prep = torch.stack(rows, dim=0)
        prep = F.pad(prep, [0, 0, 0, height - prep.shape[0]], "constant", 0)

        return prep.unsqueeze(0)

    @staticmethod
    def params_from_prep(prep: Tensor) -> List[Tensor]:
        mlp = MLP(256, 1)
        masks = list(mlp.parameters())

        prep = prep.squeeze(0)
        flatten_prep = prep.view(-1)

        params = []
        start = 0
        for mask in masks:
            end = start + mask.numel()
            params.append(flatten_prep[start:end].view(mask.shape))
            start = end

        return params

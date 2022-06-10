from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module, ABC):
    def __init__(self, net_id: int, class_id: int) -> None:
        nn.Module.__init__(self)
        ABC.__init__(self)
        self.id = net_id
        self.score = -1
        self.class_id = class_id

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def func_forward(self, x: torch.Tensor, prep: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def save(self, path: str) -> None:
        sd = self.state_dict()
        cpusd = {k: sd[k].to("cpu") for k in sd}
        torch.save(cpusd, path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def get_parameters(self) -> List[torch.Tensor]:
        return list(self.parameters())

    def get_prep(
        self,
        prep_size: Tuple[int, int],
        parameters: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        if parameters is not None:
            params = parameters
        else:
            params = self.get_parameters()

        height = prep_size[0]
        width = prep_size[1]

        flatten_params = [p.view(-1) for p in params]
        all_params = torch.cat(flatten_params, dim=0)
        param_rows = []

        start = 0
        while start < len(all_params):
            row = all_params[start : start + width]
            row = F.pad(row, [0, width - row.numel()], "constant", 0)
            param_rows.append(row)
            start += width

        prep = torch.stack(param_rows, dim=0)

        prep = F.pad(prep, [0, 0, 0, height - prep.shape[0]], "constant", 0)

        prep = prep.unsqueeze(0)

        return prep

    def get_layers(self) -> List[nn.Module]:
        return list(self.children())

    def params_from_prep(self, prep: torch.Tensor) -> List[torch.Tensor]:
        prep = prep.squeeze(0)
        flatten_prep = prep.view(-1)
        masks = self.get_parameters()
        params = []
        start = 0
        for mask in masks:
            end = start + mask.numel()
            params.append(flatten_prep[start:end].view(mask.shape))
            start = end

        return params

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

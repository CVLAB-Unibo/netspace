from typing import Callable, List, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d

from models.basenet import BaseNet
from trainers.utils import progress_bar


class SingleArchWeightsInterpolation:
    def __init__(
        self,
        ref_net: BaseNet,
        preps: Tuple[torch.Tensor, torch.Tensor],
        prep_size: Tuple[int, int],
        eval: Callable,
        device: torch.device,
    ):
        self.eval = eval
        self.device = device
        self.ref_net = ref_net
        self.preps = preps
        self.prep_size = prep_size
        self.param_mask = ref_net.get_parameters()

    @staticmethod
    def flatten_params(params: List[torch.Tensor]) -> torch.Tensor:
        flatten_params = [p.view(-1) for p in params]
        return torch.cat(flatten_params, dim=0)

    def reshape_params(self, params: torch.Tensor) -> List[torch.Tensor]:
        reshaped_params = []
        start = 0
        for mask in self.param_mask:
            end = start + mask.numel()
            reshaped_params.append(params[start:end].view(mask.shape))
            start = end

        return reshaped_params

    def interpolation_loop(self):
        flatten = SingleArchWeightsInterpolation.flatten_params
        with torch.no_grad():
            score_A = self.eval(self.ref_net, net_prep=self.preps[0])
            params_A = flatten(self.ref_net.params_from_prep(self.preps[0])).cpu().numpy()
            score_B = self.eval(self.ref_net, net_prep=self.preps[1])
            params_B = flatten(self.ref_net.params_from_prep(self.preps[1])).cpu().numpy()
            interfunc = interp1d([1, 100], np.vstack([params_A, params_B]), axis=0)

            scores = []
            scores.append(score_A)
            for i in progress_bar(range(2, 100)):
                params_i = torch.tensor(interfunc(i), dtype=torch.float)
                params_i = params_i.to(self.device)
                reshaped_params_i = self.reshape_params(params_i)
                prep_i = self.ref_net.get_prep(self.prep_size, reshaped_params_i)
                score_i = self.eval(self.ref_net, net_prep=prep_i)
                scores.append(score_i)
            scores.append(score_B)
        return scores

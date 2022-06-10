from typing import Callable, Tuple

import numpy as np
import torch
from models.basenet import BaseNet
from models.decoder import Decoder
from models.encoder import Encoder
from scipy.interpolate import interp1d
from trainers.utils import progress_bar


class SingleArchInterpolation:
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        nets: Tuple[Tuple[BaseNet, torch.Tensor]],
        eval: Callable,
        device: torch.device,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.nets = nets
        self.eval = eval
        self.device = device

    def interpolation_loop(self):
        with torch.no_grad():
            emb_A = self.encoder(self.nets[0][1].unsqueeze(0))
            prep_A = self.decoder(emb_A)[0]
            score_A = self.eval(self.nets[0][0], net_prep=prep_A)
            emb_A = emb_A.view(-1).cpu().numpy()
            emb_B = self.encoder(self.nets[1][1].unsqueeze(0))
            prep_B = self.decoder(emb_B)[0]
            score_B = self.eval(self.nets[1][0], net_prep=prep_B)
            emb_B = emb_B.view(-1).cpu().numpy()
            interfunc = interp1d([1, 100], np.vstack([emb_A, emb_B]), axis=0)

            scores = []
            scores.append(score_A)
            for i in progress_bar(range(2, 100)):
                emb_i = torch.tensor(interfunc(i), dtype=torch.float)
                emb_i = emb_i.to(self.device).view(1, -1, 1, 1)
                prep_i = self.decoder(emb_i)[0]
                score_i = self.eval(self.nets[0][0], net_prep=prep_i)
                scores.append(score_i)
            scores.append(score_B)
        return scores

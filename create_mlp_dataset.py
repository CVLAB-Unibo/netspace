from pathlib import Path
from random import sample
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from data.sdfdset import SDFDataset
from models.mlp import MLP
from trainers.utils import progress_bar


def main() -> None:
    device = "cuda"

    dataset_root = Path("/path/to/sdf/dataset")
    num_samples = 20_000

    dataset_range = [0, 1500]
    num_shapes = 500

    indices = sample(range(dataset_range[0], dataset_range[1]), num_shapes)
    dataset = SDFDataset(dataset_root, num_samples, indices)

    num_parallel_mlps = 4
    batches = []
    start = 0
    while start < num_shapes:
        end = min(start + num_parallel_mlps, num_shapes)
        batches.append(tuple(range(start, end)))
        start += num_parallel_mlps

    output_path = Path("/path/to/output/dir")
    output_path.mkdir(parents=True, exist_ok=True)

    num_steps = 10_000
    lr = 0.0001
    gamma = 0.9
    lr_sched_steps = 1000

    sdk = list(MLP(256, 1).state_dict().keys())

    for sdf_idxs in progress_bar(batches):
        params = []
        for _ in range(len(sdf_idxs)):
            mlp = MLP(256, 1).to(device)
            params.append(list(mlp.parameters()))

        batched_params = []
        for i in range(len(params[0])):
            p = torch.stack([p[i] for p in params], dim=0)
            p = torch.clone(p.detach())
            p.requires_grad = True
            batched_params.append(p)

        optimizer = Adam(batched_params, lr)
        scheduler = ExponentialLR(optimizer, gamma)

        for step in progress_bar(range(num_steps)):
            sdfs = [dataset[idx].to(device) for idx in sdf_idxs]
            sdf = torch.stack(sdfs, dim=0)
            xyz = sdf[:, :, :3]
            y = sdf[:, :, 3]

            pred = mlp_batched_forward(batched_params, xyz)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % lr_sched_steps == lr_sched_steps - 1:
                scheduler.step()

        for mlp_idx, sdf_idx in enumerate(sdf_idxs):
            sd = {k: batched_params[i][mlp_idx].detach().cpu() for i, k in enumerate(sdk)}
            name, sdf = dataset.get_sdf(sdf_idx)
            result = {"mlp": sd, "sdf": sdf}
            result_path = output_path / f"{name}.pt"
            torch.save(result, result_path)


def mlp_batched_forward(batched_params: List[Tensor], coords: Tensor) -> Tensor:
    num_layers = len(batched_params) // 2

    f = coords

    for i in range(num_layers):
        weights = batched_params[i * 2]
        biases = batched_params[i * 2 + 1]

        f = torch.bmm(f, weights.permute(0, 2, 1)) + biases.unsqueeze(1)

        if i < num_layers - 1:
            f = torch.sin(30 * f)

    return f.squeeze(-1)


if __name__ == "__main__":
    main()

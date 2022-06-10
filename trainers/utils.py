from typing import Iterable, Tuple

import numpy as np
import torch
from models.mlp import MLP
from skimage.measure import marching_cubes
from tqdm import tqdm


# from https://github.com/HobbitLong/RepDistiller
def adjust_learning_rate(epoch, lr_decay_epochs, optimizer, lr_value, lr_decay_rate=0.1):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    if lr_decay_epochs is not None:
        steps = np.sum(epoch >= np.asarray(lr_decay_epochs))
    else:
        steps = 0
    # print("current lr: ", optimizer.param_groups[0]["lr"])
    if steps > 0:
        new_lr = lr_value * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def progress_bar(iterable: Iterable, desc: str = "", ncols=60) -> Iterable:
    bar_format = "{percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    if len(desc) > 0:
        bar_format = "{desc}: " + bar_format
    return tqdm(iterable, desc=desc, bar_format=bar_format, ncols=ncols, leave=False)


@torch.no_grad()
def sdf_to_mesh(
    mlp: MLP,
    grid_size: int = 256,
    max_batch: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray]:
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (grid_size - 1)

    overall_index = torch.arange(0, grid_size ** 3, 1)
    coords = torch.zeros(grid_size ** 3, 4)

    coords[:, 2] = overall_index % grid_size
    coords[:, 1] = (overall_index.long() / grid_size) % grid_size
    coords[:, 0] = ((overall_index.long() / grid_size) / grid_size) % grid_size

    coords[:, 0] = (coords[:, 0] * voxel_size) + voxel_origin[2]
    coords[:, 1] = (coords[:, 1] * voxel_size) + voxel_origin[1]
    coords[:, 2] = (coords[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = grid_size ** 3
    start = 0

    while start < num_samples:
        end = min(start + max_batch, num_samples)
        coords_subset = coords[start:end, :3].cuda()
        sdf = mlp(coords_subset).cpu()
        coords[start:end, 3] = sdf
        start += max_batch

    sdf = coords[:, 3].reshape(grid_size, grid_size, grid_size)

    verts, faces, _, _ = marching_cubes(sdf.numpy(), level=0.0, spacing=[voxel_size] * 3)

    return verts, faces

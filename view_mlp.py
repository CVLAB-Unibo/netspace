from pathlib import Path
from random import randint

import open3d as o3d
import torch

from models.mlp import MLP
from trainers.utils import sdf_to_mesh


def main() -> None:
    dataset_path = Path("/path/to/mlp/dataset")
    mlp = MLP(256, 1).to("cuda")

    ckpt_paths = list(dataset_path.glob("*.pt"))

    stop = False
    while not stop:
        ckpt_index = randint(0, len(ckpt_paths) - 1)
        mlp_path = ckpt_paths[ckpt_index]

        ckpt = torch.load(mlp_path)

        mlp.load_state_dict(ckpt["mlp"])

        vertices, faces = sdf_to_mesh(mlp, grid_size=256)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        o3d.visualization.draw_geometries([mesh])

        s = input("Press ENTER to continue or Q+ENTER to exit: ")
        stop = s.lower() == "q"


if __name__ == "__main__":
    main()

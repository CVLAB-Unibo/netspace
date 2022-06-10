from pathlib import Path

import open3d as o3d
import torch

from data.mlpdset import MLPDataset
from models.decoder import Decoder
from models.encoder import Encoder
from models.mlp import MLP
from trainers.utils import sdf_to_mesh


@torch.no_grad()
def main() -> None:
    device = "cuda"

    num_interp = 5
    interp_step = 1 / (num_interp + 1)

    ckpt_path = Path("/path/to/netspace/ckpt")

    out_dir = Path("/out/dir")
    out_dir.mkdir(exist_ok=True)

    dataset_root = Path("/path/to/sdf/datasets")
    train_range = (0, 950)
    num_coords = 50_000

    prep_shape = (8, 10_000)
    emb_size = 4096

    mlps_train_dataset = MLPDataset(dataset_root, train_range, prep_shape, num_coords)

    encoder = Encoder(emb_size)
    decoder = Decoder([], emb_size, prep_shape, arch_prediction=False)

    ckpt = torch.load(ckpt_path)
    encoder.load_state_dict(ckpt["0"])
    encoder.eval()
    encoder.to(device)
    decoder.load_state_dict(ckpt["1"])
    decoder.eval()
    decoder.to(device)

    a, b = 0, 1000  # indices of boundaries shapes

    prep_a, _ = mlps_train_dataset[a]
    prep_b, _ = mlps_train_dataset[b]
    prep_a = prep_a.to(device)
    prep_b = prep_b.to(device)

    preps = [prep_a]
    for i in range(1, num_interp + 1):
        gamma = i * interp_step
        prep = (1 - gamma) * prep_a + gamma * prep_b
        preps.append(prep)
    preps.append(prep_b)

    embedding_a = encoder(prep_a.unsqueeze(0)).view(-1)
    embedding_b = encoder(prep_b.unsqueeze(0)).view(-1)

    embeddings = [embedding_a]
    for i in range(1, num_interp + 1):
        gamma = i * interp_step
        embedding = (1 - gamma) * embedding_a + gamma * embedding_b
        embeddings.append(embedding)
    embeddings.append(embedding_b)

    mlp = MLP(256, 1).to(device)
    sd = mlp.state_dict()

    meshes = []

    for i in range(len(preps)):
        interp_prep = preps[i]

        try:
            params = MLP.params_from_prep(interp_prep)
            sd = {k: params[i] for i, k in enumerate(sd)}
            mlp.load_state_dict(sd)
            vertices, faces = sdf_to_mesh(mlp, grid_size=256)

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            meshes.append(mesh)
        except:
            pass

    for i, mesh in enumerate(meshes):
        out_path = out_dir / f"{i}.ply"
        o3d.io.write_triangle_mesh(str(out_path), mesh)


if __name__ == "__main__":
    main()

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from data.nets import NetsDataset
from experiments.single_arch_interp import SingleArchInterpolation
from experiments.single_arch_weights_interp import SingleArchWeightsInterpolation
from models.decoder import Decoder
from models.encoder import Encoder
from models.resnet_fusedbn import ResNetFusedBN
from trainers.classification import ClassificationTrainer

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dataset = ImagesDataset(dataset_name, batch_size=128)
_, _, test_loader = dataset.get_loaders()

eval_func = partial(ClassificationTrainer.eval_accuracy, images_loader=test_loader, device=device)

input_list = f"/path/to/input/list"
ckpt_path = f"/path/to/netspace/ckpt"
save_path = f"images/{dataset_name}/single_interp.pdf"
Path(save_path).parent.mkdir(parents=True, exist_ok=True)

# prep_size = (8, 10000)
prep_size = (16, 10000)
emb_size = 4096

nets = NetsDataset(input_list, device, eval_func, prep_size)

net_A = nets[0]
net_B = nets[-1]

prep_A = net_A[0].get_prep(prep_size)
prep_B = net_B[0].get_prep(prep_size)

weights_int = SingleArchWeightsInterpolation(
    net_A[0],
    (prep_A, prep_B),
    prep_size,
    eval_func,
    device,
)

ckpt = torch.load(ckpt_path)

encoder = Encoder(emb_size)
encoder.load_state_dict(ckpt["0"])
encoder = encoder.to(device)
encoder.eval()

out_net = ResNetFusedBN(0, 2, 8, dataset_name)
decoder = Decoder([out_net], emb_size, prep_size)
decoder.load_state_dict(ckpt["1"])
decoder = decoder.to(device)
decoder.eval()

emb_int = SingleArchInterpolation(encoder, decoder, (net_A, net_B), eval_func, device)

with torch.no_grad():
    emb_A = encoder(net_A[1].unsqueeze(0))
    prep_A = decoder(emb_A)[0]
    emb_B = encoder(net_B[1].unsqueeze(0))
    prep_B = decoder(emb_B)[0]

scores = [
    weights_int.interpolation_loop(),
    emb_int.interpolation_loop(),
]

labels = ["weights interpolation", "lat. space interpolation"]

fig, ax = plt.subplots(figsize=(6, 3))
ax.set_xlabel("interpolation factor", fontsize=24)
ax.set_ylabel("accuracy", fontsize=24)
ax.grid(alpha=0.2)
ax.tick_params(axis="both", which="major", labelsize=15)

sc = scores[1]

idx = [n / 100 for n in range(len(sc))]

score_A = sc[0]
score_B = sc[-1]
ax.scatter([idx[0], idx[-1]], [score_A, score_B], marker="o", c="r", s=50, zorder=2)

ax.scatter(
    idx[1:-1],
    sc[1:-1],
    marker="+",
    c="b",
    s=50,
    label="lat. space interp.",
)

sc = scores[0]
ax.scatter(
    idx[1:-1],
    sc[1:-1],
    marker="+",
    c="fuchsia",
    s=50,
    label="weights interp.",
)

ax.legend(fontsize=18, loc="center", handletextpad=0.1)

fig.savefig(save_path, bbox_inches="tight", dpi=600)

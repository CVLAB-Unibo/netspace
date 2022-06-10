import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from data.nets import NetsDataset
from experiments.multi_arch_interp import MultiArchInterpolation
from models.decoder import Decoder
from models.encoder import Encoder
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dataset = ImagesDataset(dataset_name, batch_size=128)
_, _, test_loader = dataset.get_loaders()

eval_func = partial(ClassificationTrainer.eval_accuracy, images_loader=test_loader)

prep_size = (88, 10000)
num_archs = 4
emb_size = 4096

input_list = f"/path/to/input/list"
ckpt_path = f"/path/to/netspace/ckpt"
save_path = f"images/{dataset_name}/multi_interp.pdf"

Path(save_path).parent.mkdir(parents=True, exist_ok=True)

nets = NetsDataset(input_list, device, eval_func, prep_size)

net_A = nets[0]
net_B = nets[-1]

encoder = Encoder(emb_size)
encoder = encoder.to(device)
encoder.eval()

out_nets = []
out_nets.append(LeNetLike(0, 0, dataset_name))
out_nets.append(VanillaCNN(0, 1, dataset_name))
out_nets.append(ResNetFusedBN(0, 2, 8, dataset_name))
out_nets.append(ResNetFusedBN(0, 3, 32, dataset_name))

decoder = Decoder(out_nets, emb_size, prep_size, arch_prediction=True, num_archs=num_archs)
decoder = decoder.to(device)
decoder.eval()

ckpt = torch.load(ckpt_path)
encoder.load_state_dict(ckpt["0"])
decoder.load_state_dict(ckpt["1"])

embint = MultiArchInterpolation(encoder, decoder, (net_A, net_B), eval_func, device)

scores, archs = embint.interpolation_loop()

fig, ax = plt.subplots(figsize=(6, 3))

ax.set_xlabel("interpolation factor", fontsize=24)
ax.set_xticks([0, 0.33, 0.66, 1])
ax.set_ylabel("accuracy", fontsize=24)
ax.set_ylim(min(scores) - 5, max(scores) + 18)
ax.grid(alpha=0.2)
ax.tick_params(axis="both", which="major", labelsize=14)

color_map = {
    0: "red",
    1: "#4dadd6",
    2: "green",
    3: "fuchsia",
}

colors = [color_map[mask] for mask in archs]
idx = [n / 100 for n in range(100)]

ax.scatter(
    [idx[0], idx[-1]],
    [scores[0], scores[-1]],
    c=[colors[0], colors[-1]],
    marker="o",
    s=100,
    zorder=10,
    edgecolors="black",
)
ax.scatter(
    idx[1:-1],
    scores[1:-1],
    c=colors[1:-1],
    marker="+",
    s=100,
    zorder=2,
    label="interp. instance",
)
ax.scatter(
    [idx[33], idx[66]],
    [scores[33], scores[66]],
    c=[colors[33], colors[66]],
    marker="*",
    s=200,
    zorder=10,
    edgecolors="black",
)

ax.plot((0.33, 0.33), (-1000, scores[33]), c="black", linestyle="--", alpha=0.3, zorder=1)
ax.plot((0.66, 0.66), (-1000, scores[66]), c="black", linestyle="--", alpha=0.3, zorder=1)
ax.legend(fontsize=18, loc="upper left", handletextpad=0.1)
leg = ax.get_legend()
leg.legendHandles[0].set_color("black")

xoffset = -0.05
yoffset = 4
ax.text(0 - 0.04, scores[0] + yoffset, str(scores[0]), fontsize=16)
ax.text(0.33 + xoffset, scores[33] + yoffset, str(scores[33]), fontsize=16)
ax.text(0.66 + xoffset, scores[66] + yoffset, str(scores[66]), fontsize=16)
ax.text(1 - 0.13, scores[-1] + yoffset, str(scores[-1]), fontsize=16)

fig.savefig(save_path, bbox_inches="tight", dpi=600)

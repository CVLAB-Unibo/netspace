import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from functools import partial
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from data.nets import NetsDataset
from models.decoder import Decoder
from models.encoder import Encoder
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer
from trainers.utils import progress_bar

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME
prep_size = (88, 10000)
emb_size = 4096

input_list = f"/path/to/input/list"
ckpt_file = f"/path/to/netspace/ckpt"
save_path = f"images/{dataset_name}/multi.pdf"

Path(save_path).parent.mkdir(parents=True, exist_ok=True)

dataset = ImagesDataset(dataset_name, batch_size=128)
_, _, test_loader = dataset.get_loaders()

eval_func = partial(ClassificationTrainer.eval_accuracy, images_loader=test_loader, device=device)

nets_dataset = NetsDataset(input_list, device, eval_func, prep_size)

ckpt = torch.load(ckpt_file)

enc = Encoder(emb_size=emb_size)
enc.load_state_dict(ckpt["0"])
enc.to(device)
enc.eval()

out_nets = []
out_nets.append(LeNetLike(0, 0, dataset_name))
out_nets.append(VanillaCNN(0, 1, dataset_name))
out_nets.append(ResNetFusedBN(0, 2, 8, dataset_name))
out_nets.append(ResNetFusedBN(0, 3, 32, dataset_name))

dec = Decoder(out_nets, emb_size, prep_size, arch_prediction=True)
dec.load_state_dict(ckpt["1"])
dec.to(device)
dec.eval()

target_scores: List[float] = []
target_class_ids: List[int] = []
pred_scores: List[float] = []
pred_class_ids: List[int] = []

with torch.no_grad():
    for net, prep in progress_bar(nets_dataset):
        target_scores.append(net.score)
        target_class_ids.append(net.class_id)
        embedding = enc(prep.unsqueeze(0))
        pred_class, pred_prep = dec(embedding)
        pred_class_id = int(torch.argmax(pred_class, dim=1).item())
        pred_class_ids.append(pred_class_id)
        predicted_net = dec.out_nets[pred_class_id]
        predicted_score = eval_func(predicted_net, net_prep=pred_prep)
        pred_scores.append(predicted_score)

min_score = 1000.0
max_score = 0.0
for i in range(len(target_scores)):
    min_score = min(min_score, target_scores[i], pred_scores[i])
    max_score = max(max_score, target_scores[i], pred_scores[i])

fig, ax = plt.subplots(figsize=(6, 3))

ax.set_xlabel("target instance id", fontsize=24)
ax.set_ylabel("accuracy", fontsize=24)
ax.set_ylim(min_score - 3, max_score + 3)
ax.grid(alpha=0.2)
ax.tick_params(axis="both", which="major", labelsize=14)

color_map = {
    0: "red",
    1: "#03afff",
    2: "green",
    3: "fuchsia",
}

idx = [n for n in range(len(target_scores))]
target_colors = [color_map[arch] for arch in target_class_ids]
predicted_colors = [color_map[arch] for arch in pred_class_ids]
ax.scatter(
    idx,
    target_scores,
    c=target_colors,
    marker="o",
    s=50,
    label="target",
    zorder=2,
)
ax.scatter(
    idx,
    pred_scores,
    c=predicted_colors,
    marker="+",
    s=50,
    label="predicted",
    zorder=2,
)
ax.legend(loc="lower right", fontsize=20, handletextpad=0.1)
leg = ax.get_legend()
leg.legendHandles[0].set_color("black")
leg.legendHandles[1].set_color("black")

for i in range(len(target_scores)):
    ax.plot(
        [i, i],
        [target_scores[i], pred_scores[i]],
        linestyle=":",
        c="black",
        alpha=0.3,
        zorder=1,
    )

fig.savefig(save_path, bbox_inches="tight", dpi=600)

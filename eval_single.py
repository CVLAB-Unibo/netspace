import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from data.nets import NetsDataset
from models.decoder import Decoder
from models.encoder import Encoder
from models.resnet_fusedbn import ResNetFusedBN
from trainers.classification import ClassificationTrainer
from trainers.utils import progress_bar

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dataset = ImagesDataset(dataset_name, batch_size=128)
_, _, test_loader = dataset.get_loaders()

eval_func = partial(ClassificationTrainer.eval_accuracy, images_loader=test_loader, device=device)

test_list = f"/path/to/input/list"
ckpt_file = f"/path/to/netspace/ckpt"

out_net = ResNetFusedBN(0, 2, 8, dataset_name)

# prep_size = (8, 10000)
prep_size = (16, 10000)
emb_size = 4096

save_path = f"images/{dataset_name}/single.pdf"
Path(save_path).parent.mkdir(parents=True, exist_ok=True)

nets_test_dataset = NetsDataset(test_list, device, eval_func, prep_size)

ckpt = torch.load(ckpt_file)

enc = Encoder(emb_size=emb_size)
enc.load_state_dict(ckpt["0"])
enc.to(device)
enc.eval()

dec = Decoder([out_net], emb_size, prep_size)
dec.load_state_dict(ckpt["1"])
dec.to(device)
dec.eval()

target_scores = []
predicted_scores = []

with torch.no_grad():
    for i in progress_bar(range(len(nets_test_dataset))):
        net, prep = nets_test_dataset[i]
        target_scores.append(net.score)
        embedding = enc(prep.unsqueeze(0))
        predicted_prep = dec(embedding)[0]
        predicted_score = eval_func(net, net_prep=predicted_prep)
        predicted_scores.append(predicted_score)

min_score = 1000.0
max_score = 0.0
for i in range(len(target_scores)):
    min_score = min(min_score, target_scores[i], predicted_scores[i])
    max_score = max(max_score, target_scores[i], predicted_scores[i])

fig, ax = plt.subplots(figsize=(6, 3))

ax.set_xlabel("target instance id", fontsize=24)
ax.set_ylabel("accuracy", fontsize=24)
ax.set_ylim(min_score - 3, max_score + 3)
ax.grid(alpha=0.2)
ax.tick_params(axis="both", which="major", labelsize=15)

idx = [n for n in range(len(target_scores))]
ax.scatter(idx, target_scores, c="r", marker="o", s=50, label="target", zorder=2)
ax.scatter(idx, predicted_scores, c="b", marker="+", s=50, label="predicted", zorder=2)
ax.legend(fontsize=18, loc="lower right", handletextpad=0.1)

for i in range(len(target_scores)):
    ax.plot(
        [i, i],
        [target_scores[i], predicted_scores[i]],
        linestyle=":",
        c="black",
        alpha=0.3,
        zorder=1,
    )

fig.savefig(save_path, bbox_inches="tight", dpi=600)

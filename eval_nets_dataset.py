from pathlib import Path

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer
from trainers.utils import progress_bar

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dset = ImagesDataset(dataset_name)
_, loader, _ = dset.get_loaders()

input_list = f"/path/to/input/list"

with open(input_list, "r") as f:
    lines = [line.strip() for line in f.readlines()]

ckpt_paths = []

for line in lines:
    path = Path(line.split(";")[-1])
    ckpt_paths.append(path)

accs = []

for ckpt in progress_bar(ckpt_paths):
    # net = LeNetLike(0, 0, dataset_name)
    # net = VanillaCNN(0, 1, dataset_name)
    # net = ResNetFusedBN(0, 2, 8, dataset_name)
    net = ResNetFusedBN(0, 3, 32, dataset_name)

    net.load(ckpt.absolute())
    net = net.to("cuda")

    acc = ClassificationTrainer.eval_accuracy(net, loader)
    accs.append((acc, ckpt))

accs.sort(key=lambda x: x[0])
print(accs)

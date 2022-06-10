from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from data.nets import NetsDataset
from models.decoder import Decoder
from models.encoder import Encoder
from models.resnet_fusedbn import ResNetFusedBN
from trainers.classification import ClassificationTrainer
from trainers.single_arch import SingleArchTrainer

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

# prep_size = (8, 10_000)
prep_size = (16, 10_000)
emb_size = 4096

net_batch_size = 8
images_batch_size = 128
epoch_num = 1000
lr = 0.0001

dataset = ImagesDataset(dataset_name, batch_size=images_batch_size)

train_input_list = f"/path/to/train/input/list"
val_input_list = f"/path/to/val/input/list"
logdir = f"logs/{dataset_name}/single/resnet8"

Path(logdir).mkdir(parents=True, exist_ok=True)

train_loader, val_loader, _ = dataset.get_loaders()

eval_func = partial(ClassificationTrainer.eval_accuracy, images_loader=val_loader)
nets_train_dataset = NetsDataset(train_input_list, device, eval_func, prep_size)
nets_val_dataset = NetsDataset(val_input_list, device, eval_func, prep_size)

nets_train_dataloader = DataLoader(
    nets_train_dataset,
    batch_size=net_batch_size,
    collate_fn=NetsDataset.collate_fn,
    shuffle=True,
)
nets_val_dataloader = DataLoader(
    nets_val_dataset,
    batch_size=net_batch_size,
    collate_fn=NetsDataset.collate_fn,
    shuffle=False,
)

encoder = Encoder(emb_size).to(device)
out_nets = [ResNetFusedBN(0, 2, 8, dataset_name)]
decoder = Decoder(out_nets, emb_size, prep_size, arch_prediction=False).to(device)

trainer = SingleArchTrainer(device, logdir)
trainer.train(
    encoder,
    decoder,
    nets_train_dataloader,
    nets_val_dataloader,
    train_loader,
    val_loader,
    epoch_num,
)

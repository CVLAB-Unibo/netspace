from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from data.nets import NetsDataset
from data.nets_missing import NetsDatasetMissing
from models.decoder import Decoder
from models.encoder import Encoder
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer
from trainers.multi_arch import MultiArchTrainer

device = torch.device("cuda")

dataset_name = TINY_IMAGENET_NAME
# dataset_name = CIFAR10_NAME

epoch_num = 1000
num_archs = 4
prep_size = (88, 10_000)
emb_size = 4096

logdir = f"logs/{dataset_name}/multi"
train_input_list = f"/path/to/train/input/list"
val_input_list = f"/path/to/val/input/list"
net_batch_size = 2
missing = False

# logdir = f"logs/{dataset_name}/missing"
# train_input_list = f"/path/to/train/input/list"
# val_input_list = f"/path/to/val/input/list"
# net_batch_size = 1
# missing = True

Path(logdir).mkdir(parents=True, exist_ok=True)

dataset = ImagesDataset(dataset_name, batch_size=128)
train_loader, val_loader, _ = dataset.get_loaders()

teacher_net = ResNetFusedBN(0, 4, 56, dataset_name)
teacher_net.load(f"/path/to/teacher/network/ckpt")
teacher_net.to(device)

eval_func = partial(ClassificationTrainer.eval_accuracy, images_loader=val_loader, device=device)

nets_dataset = NetsDataset if not missing else NetsDatasetMissing

nets_train_dataset = nets_dataset(train_input_list, device, eval_func, prep_size)
nets_train_loader = DataLoader(
    nets_train_dataset,
    batch_size=net_batch_size,
    shuffle=True,
    collate_fn=nets_dataset.collate_fn,
)
nets_val_dataset = nets_dataset(val_input_list, device, eval_func, prep_size)
nets_val_loader = DataLoader(
    nets_val_dataset,
    batch_size=net_batch_size,
    collate_fn=nets_dataset.collate_fn,
)

encoder = Encoder(emb_size=emb_size)
encoder.to(device)

out_nets = []
out_nets.append(LeNetLike(0, 0, dataset_name))
out_nets.append(VanillaCNN(0, 1, dataset_name))
out_nets.append(ResNetFusedBN(0, 2, 8, dataset_name))
out_nets.append(ResNetFusedBN(0, 3, 32, dataset_name))

decoder = Decoder(out_nets, emb_size, prep_size, arch_prediction=True, num_archs=num_archs)
decoder.to(device)

trainer = MultiArchTrainer(device, logdir)
trainer.train(
    encoder,
    decoder,
    nets_train_loader,
    nets_val_loader,
    train_loader,
    val_loader,
    epoch_num,
    num_archs,
    teacher_net,
)

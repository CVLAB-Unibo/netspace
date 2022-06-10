from functools import partial
from pathlib import Path

from torch.optim import SGD

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.lenetlike import LeNetLike
from models.resnet import ResNet
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer
from trainers.utils import progress_bar

device = "cuda"
trainer_id = 0
repeat = 2

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dataset = ImagesDataset(dataset_name, batch_size=128)
train_loader, val_loader, _ = dataset.get_loaders()

save_path = Path(f"models/ckpts/{dataset_name}/multi/")
save_path.mkdir(parents=True, exist_ok=True)

builders = {
    "lenet": partial(LeNetLike, net_id=0, class_id=0, dataset_name=dataset_name),
    "vanillacnn": partial(VanillaCNN, net_id=0, class_id=1, dataset_name=dataset_name),
    "resnet8": partial(ResNet, net_id=0, class_id=2, depth=8, dataset_name=dataset_name),
    "resnet32": partial(ResNet, net_id=0, class_id=3, depth=32, dataset_name=dataset_name),
}

num_nets = {
    "lenet": 6,
    "vanillacnn": 5,
    "resnet8": 6,
    "resnet32": 10,
}

ids = {
    "lenet": trainer_id * repeat * num_nets["lenet"],
    "vanillacnn": trainer_id * repeat * num_nets["vanillacnn"],
    "resnet8": trainer_id * repeat * num_nets["resnet8"],
    "resnet32": trainer_id * repeat * num_nets["resnet32"],
}

nets = []
nets.extend(["lenet"] * num_nets["lenet"] * repeat)
nets.extend(["vanillacnn"] * num_nets["vanillacnn"] * repeat)
nets.extend(["resnet8"] * num_nets["resnet8"] * repeat)
nets.extend(["resnet32"] * num_nets["resnet32"] * repeat)

for net_name in progress_bar(nets, f"Trainer {trainer_id}"):
    net = builders[net_name]()
    net = net.to(device)

    lr = 0.05
    optimizer = SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    epoch_num = 300
    lr_decay_epochs = [149, 179, 209]

    ckpt_path = save_path / net_name / f"{ids[net_name]}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    tasktrainer = ClassificationTrainer(device, None, ckpt_path.absolute(), "best")

    tasktrainer.train(
        net,
        train_loader,
        val_loader=val_loader,
        epoch_num=epoch_num,
        optimizer=optimizer,
        lr_value=lr,
        lr_decay_epochs=lr_decay_epochs,
    )

    ids[net_name] += 1

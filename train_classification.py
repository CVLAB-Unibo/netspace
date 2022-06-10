from pathlib import Path

from torch.optim import SGD

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.lenetlike import LeNetLike
from models.resnet import ResNet
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer

device = "cuda"

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dataset = ImagesDataset(dataset_name, batch_size=128)
train_loader, val_loader, test_loader = dataset.get_loaders()

net = LeNetLike(0, 0, dataset_name)
# net = VanillaCNN(0, 1, dataset_name)
# net = ResNet(0, 2, 8, dataset_name)
# net = ResNet(0, 3, 32, dataset_name)

net = net.to(device)

logdir = Path(f"logs/{dataset_name}/lenet")
logdir.mkdir(parents=True)
ckpt_path = logdir / "ckpt.pt"
trainer = ClassificationTrainer(device, logdir, ckpt_path, "final", show_progress=True)

lr = 0.05
optimizer = SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
epoch_num = 300
lr_decay_epochs = [149, 179, 209]

trainer.train(
    net,
    train_loader,
    val_loader=val_loader,
    epoch_num=epoch_num,
    optimizer=optimizer,
    lr_value=lr,
    lr_decay_epochs=lr_decay_epochs,
)

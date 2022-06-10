from functools import partial
from pathlib import Path

from torch.optim import Adam

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.lenetlike import LeNetLike
from models.resnet import ResNet
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer
from trainers.utils import progress_bar

device = "cuda"

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME
net_name = "resnet8"

dataset = ImagesDataset(dataset_name, batch_size=128)
train_loader, val_loader, _ = dataset.get_loaders()

save_path = Path(f"models/ckpts/{dataset_name}/single")
save_path.mkdir(parents=True, exist_ok=True)

builders = {
    "lenet": partial(LeNetLike, net_id=0, class_id=0, dataset_name=dataset_name),
    "vanillacnn": partial(VanillaCNN, net_id=0, class_id=1, dataset_name=dataset_name),
    "resnet8": partial(ResNet, net_id=0, class_id=2, depth=8, dataset_name=dataset_name),
    "resnet32": partial(ResNet, net_id=0, class_id=3, depth=32, dataset_name=dataset_name),
}
builder = builders[net_name]

epochs = []
for epoch in range(0, 600):
    if epoch < 100:
        freq = 1
    elif epoch < 200:
        freq = 2
    elif epoch < 300:
        freq = 5
    elif epoch < 400:
        freq = 10
    elif epoch < 500:
        freq = 25
    elif epoch < 600:
        freq = 50
    if epoch % freq == freq - 1:
        epochs.append(epoch)

start = 0
end = 46

for epoch in progress_bar(epochs[start:end]):
    net = builder()
    net = net.to(device)

    lr = 0.0001
    optimizer = Adam(net.parameters(), lr=lr)
    lr_decay_epochs = None

    ckpt_path = save_path / net_name / f"{epoch}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    tasktrainer = ClassificationTrainer(device, None, ckpt_path.absolute(), "final")

    tasktrainer.train(
        net,
        train_loader,
        val_loader=val_loader,
        epoch_num=epoch,
        optimizer=optimizer,
        lr_value=lr,
        lr_decay_epochs=lr_decay_epochs,
    )

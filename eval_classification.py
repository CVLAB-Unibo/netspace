from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dset = ImagesDataset(dataset_name)
_, loader, _ = dset.get_loaders()

# net = LeNetLike(0, 0, dataset_name)
# net.load(f"models/ckpts/{dataset_name}/multi/lenet/best.pt")
# net = net.to("cuda")

# net = VanillaCNN(0, 1, dataset_name)
# net.load(f"models/ckpts/{dataset_name}/multi/vanillacnn/best.pt")
# net = net.to("cuda")

# net = ResNetFusedBN(0, 2, 8, dataset_name)
# net.load(f"models/ckpts/{dataset_name}/multi/resnet8/best.pt")
# net = net.to("cuda")

net = ResNetFusedBN(0, 3, 32, dataset_name)
net.load(f"models/ckpts/{dataset_name}/multi/resnet32/best.pt")
net = net.to("cuda")

acc = ClassificationTrainer.eval_accuracy(net, loader)
print(acc)

import torch

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.decoder import Decoder
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from trainers.classification import ClassificationTrainer

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dataset = ImagesDataset(dataset_name)

decoder_ckpt_path = f"/path/to/decoder/ckpt"
embedding_path = f"/path/to/embeddings"

net = ResNetFusedBN(0, 2, 8, dataset_name)
net.eval()

prep_size = (8, 10000) if dataset_name == CIFAR10_NAME else (16, 10000)
emb_size = 4096
_, _, test_loader = dataset.get_loaders()

decoder = Decoder(None, emb_size, prep_size)
ckpt = torch.load(decoder_ckpt_path)
decoder.load_state_dict(ckpt["1"])
decoder = decoder.to(device)
decoder.eval()

embedding = torch.load(embedding_path)
pred_prep = decoder(embedding)[0]

test_accuracy = ClassificationTrainer.eval_accuracy(net, test_loader, pred_prep, device)

print("Test accuracy:", test_accuracy)

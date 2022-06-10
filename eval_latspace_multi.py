import torch

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.decoder import Decoder
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from models.vanillacnn import VanillaCNN
from trainers.classification import ClassificationTrainer

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME

dataset = ImagesDataset(dataset_name)

decoder_ckpt_path = f"/path/to/decoder/ckpt"
embedding_path = f"/path/to/embedding"

net = LeNetLike(0, 0, dataset_name)
# net = VanillaCNN(0, 1, dataset_name)
# net = ResNetFusedBN(0, 2, 8, dataset_name)
# net = ResNetFusedBN(0, 3, 32, dataset_name)

net.eval()

prep_size = (88, 10000)
emb_size = 4096
_, _, test_loader = dataset.get_loaders()

out_nets = [
    LeNetLike(0, 0, dataset_name),
    VanillaCNN(0, 1, dataset_name),
    ResNetFusedBN(0, 2, 8, dataset_name),
    ResNetFusedBN(0, 3, 32, dataset_name),
]
decoder = Decoder(out_nets, emb_size, prep_size, arch_prediction=True, num_archs=4)
ckpt = torch.load(decoder_ckpt_path)
decoder.load_state_dict(ckpt["1"])
decoder = decoder.to(device)
decoder.eval()

embedding = torch.load(embedding_path)
pred_class, pred_prep = decoder(embedding)

pred_class_id = torch.argmax(pred_class, dim=1).item()

if pred_class_id == net.class_id:
    print("The predicted architecture is correct.")
else:
    print("The predicted architecture differs from the specified one.")

pred_arch = decoder.out_nets[pred_class_id]

test_accuracy = ClassificationTrainer.eval_accuracy(pred_arch, test_loader, pred_prep, device)

print("Test accuracy:", test_accuracy)

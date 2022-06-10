import torch

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.decoder import Decoder
from models.encoder import Encoder
from models.lenetlike import LeNetLike
from models.resnet_fusedbn import ResNetFusedBN
from models.vanillacnn import VanillaCNN
from trainers.latspace_multi import LatSpaceTrainer

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME
channels = ImagesDataset(dataset_name).channels

decoder_ckpt_path = f"/path/to/netspace/ckpt"

net = LeNetLike(0, 0, dataset_name)
logdir = f"logs/{dataset_name}/latspace_multi_lenet"
# net = VanillaCNN(0, 1, dataset_name)
# net = ResNetFusedBN(0, 2, 8, dataset_name)
# net = ResNetFusedBN(0, 3, 32, dataset_name)

ref_net_ckpt = f"/path/to/reference/network/ckpt"
ref_net = LeNetLike(0, 0, dataset_name)
# ref_net = VanillaCNN(0, 1, dataset_name)
# ref_net = ResNetFusedBN(0, 2, 8, dataset_name)
# ref_net = ResNetFusedBN(0, 3, 32, dataset_name)
ref_net.load(ref_net_ckpt)
ref_net = ref_net.to(device)
ref_net.eval()

emb_size = 4096
prep_size = (88, 10000)
num_epochs = 500
lr = 0.0001

out_nets = [
    LeNetLike(0, 0, dataset_name),
    VanillaCNN(0, 1, dataset_name),
    ResNetFusedBN(0, 2, 8, dataset_name),
    ResNetFusedBN(0, 3, 32, dataset_name),
]

encoder = Encoder(emb_size)
decoder = Decoder(out_nets, emb_size, prep_size, arch_prediction=True, num_archs=4)
ckpt = torch.load(decoder_ckpt_path)

encoder.load_state_dict(ckpt["0"])
encoder = encoder.to(device)
encoder.eval()

decoder.load_state_dict(ckpt["1"])
decoder = decoder.to(device)
decoder.eval()

teacher_net_ckpt = f"/path/to/teacher/network/ckpt"
teacher_net = ResNetFusedBN(0, 0, 56, dataset_name)
teacher_net.load(teacher_net_ckpt)
teacher_net = teacher_net.to(device)
teacher_net.eval()

with torch.no_grad():
    ref_prep = ref_net.get_prep(prep_size)
    ref_emb = encoder(ref_prep.unsqueeze(0))

trainer = LatSpaceTrainer(
    dataset_name,
    decoder,
    emb_size,
    device,
    logdir,
    net,
    teacher_net,
    lr,
    ref_emb,
)
trainer.train(num_epochs)

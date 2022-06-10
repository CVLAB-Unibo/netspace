from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME, ImagesDataset
from models.decoder import Decoder
from models.encoder import Encoder
from models.resnet_fusedbn import ResNetFusedBN

device = torch.device("cuda")

# dataset_name = CIFAR10_NAME
dataset_name = TINY_IMAGENET_NAME
channels = ImagesDataset(dataset_name).channels

# prep_size_single = (8, 10000)
prep_size_single = (16, 10000)
prep_size_multi = (88, 10000)
emb_size = 4096

ckpt_single_path = f"/path/to/netspace/single-arch/ckpt"
ckpt_multi_path = f"/path/to/netspace/multi-arch/ckpt"
save_path = f"images/{dataset_name}"

Path(save_path).mkdir(parents=True, exist_ok=True)

ckpt_single = torch.load(ckpt_single_path)
ckpt_multi = torch.load(ckpt_multi_path)

enc_single = Encoder(emb_size)
enc_single.load_state_dict(ckpt_single["0"])
enc_single = enc_single.to(device)
enc_single.eval()

enc_multi = Encoder(emb_size)
enc_multi.load_state_dict(ckpt_multi["0"])
enc_multi = enc_multi.to(device)
enc_multi.eval()

dec_single = Decoder([], emb_size, prep_size_single)
dec_single.load_state_dict(ckpt_single["1"])
dec_single = dec_single.to(device)
dec_single.eval()

dec_multi = Decoder([], emb_size, prep_size_multi, arch_prediction=True)
dec_multi.load_state_dict(ckpt_multi["1"])
dec_multi = dec_multi.to(device)
dec_multi.eval()

resnet8_ckpt_path = f"/path/to/resnet8/ckpt"
resnet8 = ResNetFusedBN(0, 2, 8, dataset_name)
resnet8.load(resnet8_ckpt_path)
resnet8 = resnet8.to(device)

resnet32_ckpt_path = f"/path/to/resnet32/ckpt"
resnet32 = ResNetFusedBN(0, 3, 32, dataset_name)
resnet32.load(resnet32_ckpt_path)
resnet32 = resnet32.to(device)

resnet8_original_prep = resnet8.get_prep(prep_size_single)
with torch.no_grad():
    resnet8_embedding = enc_single(resnet8.get_prep(prep_size_single).unsqueeze(0))
    resnet8_predicted_prep = dec_single(resnet8_embedding)

resnet32_original_prep = resnet32.get_prep(prep_size_multi)
with torch.no_grad():
    resnet32_embedding = enc_multi(resnet32.get_prep(prep_size_multi).unsqueeze(0))
    _, resnet32_predicted_prep = dec_multi(resnet32_embedding)

resnet8_num_parameters = resnet8.num_parameters()
# resnet8_img_size = (279, 279)
resnet8_img_size = (359, 359)

resnet8_images = []
for prep in [resnet8_original_prep, resnet8_predicted_prep]:
    parameters = prep.view(-1)[:resnet8_num_parameters]
    required_num_parameters = resnet8_img_size[0] * resnet8_img_size[1]
    padding = [0, required_num_parameters - resnet8_num_parameters]
    padded_parameters = F.pad(parameters, padding, "constant", 0)
    img = padded_parameters.view(resnet8_img_size)
    resnet8_images.append(img)

resnet32_num_parameters = resnet32.num_parameters()
# resnet32_img_size = (683, 683)
resnet32_img_size = (719, 719)

resnet32_images = []
for prep in [resnet32_original_prep, resnet32_predicted_prep]:
    parameters = prep.view(-1)[:resnet32_num_parameters]
    required_num_parameters = resnet32_img_size[0] * resnet32_img_size[1]
    padding = [0, required_num_parameters - resnet32_num_parameters]
    padded_parameters = F.pad(parameters, padding, "constant", 0)
    img = padded_parameters.view(resnet32_img_size)
    resnet32_images.append(img)

resnet8_img = torch.cat(resnet8_images, dim=1).detach().cpu().numpy()
resnet8_min = np.percentile(resnet8_img, 1)
resnet8_max = np.percentile(resnet8_img, 99)

fig = plt.figure(figsize=(14, 5))

for i in range(1, 3):
    img = resnet8_images[i - 1]
    fig.add_subplot(1, 2, i)
    plt.axis("off")
    im = plt.imshow(img.detach().cpu().numpy(), cmap="plasma", vmin=resnet8_min, vmax=resnet8_max)
    plt.colorbar(im)

plt.savefig(save_path + "/prep_resnet8.pdf", bbox_inches="tight", dpi=300)

resnet32_img = torch.cat(resnet32_images, dim=1).detach().cpu().numpy()
resnet32_min = np.percentile(resnet32_img, 1)
resnet32_max = np.percentile(resnet32_img, 99)

fig = plt.figure(figsize=(14, 5))

for i in range(1, 3):
    img = resnet32_images[i - 1]
    fig.add_subplot(1, 2, i)
    plt.axis("off")
    im = plt.imshow(img.detach().cpu().numpy(), cmap="plasma", vmin=resnet8_min, vmax=resnet8_max)
    plt.colorbar(im)

plt.savefig(save_path + "/prep_resnet32.pdf", bbox_inches="tight", dpi=300)

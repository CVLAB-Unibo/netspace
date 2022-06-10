import torch.nn.functional as F
from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME
from torch import Tensor, nn

from models.basenet import BaseNet


class VanillaCNN(BaseNet):
    def __init__(self, net_id: int, class_id: int, dataset_name: str) -> None:
        super(VanillaCNN, self).__init__(net_id, class_id)

        assert dataset_name in [TINY_IMAGENET_NAME, CIFAR10_NAME]

        if dataset_name == TINY_IMAGENET_NAME:
            config = [[8, 3, 2], [16, 3, 2], [32, 3, 2], [64, 3, 2], [64, 3, 2]]
            input_size = 64
            in_ch = 3
            num_classes = 200
        else:
            config = [[8, 3, 2], [8, 3, 1], [32, 3, 2], [64, 3, 1], [64, 3, 2]]
            input_size = 32
            in_ch = 3
            num_classes = 10

        tot_stride = 1
        self.layers = nn.ModuleList()
        for conf in config:
            out_ch, ks, stride = conf
            self.layers.append(nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=1))
            in_ch = out_ch
            tot_stride *= stride

        final_size = input_size // tot_stride
        out_ch = config[-1][0]
        self.layers.append(nn.Linear(final_size * final_size * out_ch, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x))
        x = x.view(x.size()[0], -1)
        x = self.layers[-1](x)

        return x

    def func_forward(self, x: Tensor, prep: Tensor) -> Tensor:
        params = self.params_from_prep(prep)

        for i in range(len(self.layers[:-1])):
            stride = self.layers[i].stride
            idx = i * 2
            x = F.conv2d(x, params[idx], bias=params[idx + 1], stride=stride, padding=1)
            x = F.leaky_relu(x)

        x = x.view(x.size()[0], -1)
        x = F.linear(x, params[-2], bias=params[-1])

        return x

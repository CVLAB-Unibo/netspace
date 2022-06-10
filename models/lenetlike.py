import torch
import torch.nn as nn
import torch.nn.functional as F
from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME

from models.basenet import BaseNet


class LeNetLike(BaseNet):
    def __init__(self, net_id: int, class_id: int, dataset_name: str) -> None:
        super(LeNetLike, self).__init__(net_id, class_id)

        assert dataset_name in [CIFAR10_NAME, TINY_IMAGENET_NAME]

        if dataset_name == TINY_IMAGENET_NAME:
            self.build_tin()
            self.forward_fn = self.forward_tin
            self.func_forward_fn = self.func_forward_tin
        else:
            self.build_cifar10()
            self.forward_fn = self.forward_cifar10
            self.func_forward_fn = self.func_forward_cifar10

    def build_tin(self) -> None:
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 200)

    def forward_tin(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def func_forward_tin(self, x: torch.Tensor, prep: torch.Tensor) -> torch.Tensor:
        params = self.params_from_prep(prep)

        x = F.leaky_relu(F.conv2d(x, params[0], bias=params[1], stride=2, padding=2))
        x = F.leaky_relu(F.conv2d(x, params[2], bias=params[3], stride=2))
        x = F.leaky_relu(F.conv2d(x, params[4], bias=params[5], stride=2))
        x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(F.linear(x, params[6], bias=params[7]))
        x = F.leaky_relu(F.linear(x, params[8], bias=params[9]))
        x = F.linear(x, params[10], bias=params[11])

        return x

    def build_cifar10(self) -> None:
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward_cifar10(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def func_forward_cifar10(self, x: torch.Tensor, prep: torch.Tensor) -> torch.Tensor:
        params = self.params_from_prep(prep)

        x = F.leaky_relu(F.conv2d(x, params[0], bias=params[1], stride=2))
        x = F.leaky_relu(F.conv2d(x, params[2], bias=params[3], stride=2))
        x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(F.linear(x, params[4], bias=params[5]))
        x = F.leaky_relu(F.linear(x, params[6], bias=params[7]))
        x = F.linear(x, params[8], bias=params[9])

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_fn(x)

    def func_forward(self, x: torch.Tensor, prep: torch.Tensor) -> torch.Tensor:
        return self.func_forward_fn(x, prep)

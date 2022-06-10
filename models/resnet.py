import torch
from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME
from torch import nn
from torchvision.models.resnet import BasicBlock as TorchBasicBlock
from torchvision.models.resnet import ResNet as TorchResNet

from models.basenet import BaseNet


class BasicBlock(TorchBasicBlock):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        TorchBasicBlock.__init__(
            self, inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return TorchBasicBlock.forward(self, x)

    def get_parameters(self):
        parameters = []
        convs = [self.conv1, self.conv2]
        bns = [self.bn1, self.bn2]
        if self.downsample:
            convs.append(self.downsample[0])
            bns.append(self.downsample[1])

        for i in range(len(convs)):
            parameters.append(convs[i].weight)
            parameters.append(bns[i].running_mean)
            parameters.append(bns[i].running_var)
            parameters.append(bns[i].weight)
            parameters.append(bns[i].bias)

        return parameters


class ResNet(BaseNet, TorchResNet):
    def __init__(self, net_id: int, class_id: int, depth: int, dataset_name: str) -> None:
        BaseNet.__init__(self, net_id, class_id)

        assert dataset_name in [TINY_IMAGENET_NAME, CIFAR10_NAME]

        if dataset_name == TINY_IMAGENET_NAME:
            in_ch = 3
            num_classes = 200
            out_mul = 4
        else:
            in_ch = 3
            num_classes = 10
            out_mul = 1

        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        error_msg = "Depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
        assert (depth - 2) % 6 == 0, error_msg
        n = (depth - 2) // 6
        block = BasicBlock

        self._norm_layer = nn.BatchNorm2d

        num_filters = [16, 16, 32, 64]

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(in_ch, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        # H/2 x W/2

        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        # H/4 x W/4
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        # H/8 x H/8
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(num_filters[3] * block.expansion * out_mul, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def func_forward(self, x, prep):
        raise NotImplementedError

    def get_parameters(self):
        parameters = []

        parameters.append(self.conv1.weight)
        parameters.append(self.bn1.running_mean)
        parameters.append(self.bn1.running_var)
        parameters.append(self.bn1.weight)
        parameters.append(self.bn1.bias)

        layers = [self.layer1, self.layer2, self.layer3]
        for layer in layers:
            for block in layer:
                parameters.extend(block.get_parameters())

        parameters.append(self.fc.weight)
        parameters.append(self.fc.bias)

        return parameters

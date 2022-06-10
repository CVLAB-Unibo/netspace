import torch
import torch.nn.functional as F
from data.images import CIFAR10_NAME, TINY_IMAGENET_NAME
from torch import nn

from models.basenet import BaseNet
from models.resnet import ResNet


def fuse_conv_and_bn(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True
    )
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_block_convs_and_bns(block):
    fused_block = BasicBlock()
    fused_block.conv1 = fuse_conv_and_bn(block.conv1, block.bn1)
    fused_block.conv2 = fuse_conv_and_bn(block.conv2, block.bn2)
    if block.downsample is not None:
        fused_block.downsample = fuse_conv_and_bn(block.downsample[0], block.downsample[1])

    return fused_block


def get_conv_like(conv):
    return nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=True,
    )


def get_block_like(block):
    new_block = BasicBlock()
    new_block.conv1 = get_conv_like(block.conv1)
    new_block.conv2 = get_conv_like(block.conv2)
    if block.downsample is not None:
        new_block.downsample = get_conv_like(block.downsample[0])
    return new_block


class BasicBlock(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = None
        self.conv2 = None
        self.downsample = None
        self.activation = F.leaky_relu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

    def func_forward(self, x, params):
        identity = x

        out = F.conv2d(x, params[0], bias=params[1], padding=1, stride=self.conv1.stride)
        out = self.activation(out)

        out = F.conv2d(out, params[2], bias=params[3], padding=1)

        if self.downsample is not None:
            identity = F.conv2d(x, params[4], bias=params[5], stride=self.downsample.stride)

        out += identity
        out = self.activation(out)

        return out

    def get_parameters(self):
        parameters = []
        convs = [self.conv1, self.conv2]
        if self.downsample:
            convs.append(self.downsample)

        for conv in convs:
            parameters.append(conv.weight)
            parameters.append(conv.bias)

        return parameters


class ResNetFusedBN(BaseNet):
    def __init__(self, net_id, class_id, depth, dataset_name):
        BaseNet.__init__(self, net_id, class_id)

        assert dataset_name in [TINY_IMAGENET_NAME, CIFAR10_NAME]
        self.dataset_name = dataset_name
        self.in_ch = 3

        self.depth = depth
        self.activation = F.leaky_relu

        ref_net = ResNet(0, 0, depth, dataset_name)
        self.init_like(ref_net)

    def load(self, path: str) -> None:
        ref_net = ResNet(0, 0, self.depth, self.dataset_name)
        ref_net.eval()
        ref_net.load(path)
        self.init_from_ref_net(ref_net)

    def init_from_ref_net(self, ref_net):
        self.conv1 = fuse_conv_and_bn(ref_net.conv1, ref_net.bn1)

        ref_layers = [ref_net.layer1, ref_net.layer2, ref_net.layer3]
        self.layers = nn.ModuleList()
        for layer in ref_layers:
            blocks = []
            for block in layer:
                fused_block = fuse_block_convs_and_bns(block)
                fused_block.activation = self.activation
                blocks.append(fused_block)
            self.layers.append(nn.Sequential(*blocks))

        self.avgpool = nn.AvgPool2d(8)
        self.fc = ref_net.fc

    def init_like(self, ref_net):
        self.conv1 = get_conv_like(ref_net.conv1)

        ref_layers = [ref_net.layer1, ref_net.layer2, ref_net.layer3]
        self.layers = nn.ModuleList()
        for layer in ref_layers:
            blocks = []
            for block in layer:
                new_block = get_block_like(block)
                new_block.activation = self.activation
                blocks.append(new_block)
            self.layers.append(nn.Sequential(*blocks))

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(ref_net.fc.weight.shape[1], ref_net.fc.weight.shape[0], bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def func_forward(self, x, prep):
        params = self.params_from_prep(prep)

        x = F.conv2d(x, params[0], bias=params[1], stride=1, padding=1)
        x = self.activation(x)

        params_start = 2

        for layer in self.layers:
            for block in layer:
                params_end = params_start + len(block.get_parameters())
                x = block.func_forward(x, params[params_start:params_end])
                params_start = params_end

        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        x = F.linear(x, params[params_start], bias=params[params_start + 1])

        return x

    def get_parameters(self):
        parameters = []

        parameters.append(self.conv1.weight)
        parameters.append(self.conv1.bias)

        for layer in self.layers:
            for block in layer:
                parameters.extend(block.get_parameters())

        parameters.append(self.fc.weight)
        parameters.append(self.fc.bias)

        return parameters

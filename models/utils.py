import torch
from torch import nn


def conv_block(
    in_ch,
    out_ch,
    ks,
    stride=1,
    padding=0,
    bn=False,
    init=False,
    bias=True,
):
    modules = []

    conv = nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=padding, bias=bias)
    if init:
        nn.init.xavier_uniform_(conv)
    modules.append(conv)

    if bn:
        modules.append(nn.BatchNorm2d(out_ch))

    block = nn.Sequential(*modules)

    return block


class ScaleLayer(nn.Module):
    def __init__(self, size, init_value_w=1e-3, init_value_b=1e-2):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(size) * init_value_w)
        self.bias = nn.Parameter(torch.rand(size) * init_value_b)

    def forward(self, x):
        return x * self.scale + self.bias

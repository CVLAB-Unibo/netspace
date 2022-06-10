import torch.nn.functional as F
from torch import nn

from models.utils import ScaleLayer, conv_block


class Decoder(nn.Module):
    def __init__(self, out_nets, emb_size, prep_size, arch_prediction=False, num_archs=4):
        super(Decoder, self).__init__()

        self.out_nets = out_nets
        self.prep_height = prep_size[0]
        self.prep_width = prep_size[1]
        self.arch_prediction = arch_prediction

        if self.arch_prediction:
            self.arch_conv = conv_block(1, 10, 3, stride=2, padding=1)
            self.arch_classifier = nn.Linear(10 * emb_size // 4, num_archs)

        self.in_ch = emb_size // 4096

        self.conv1 = conv_block(self.in_ch, self.prep_height * 256, 3, stride=2, padding=1)
        self.conv2 = conv_block(32 * 4, self.prep_width // 8, 3, stride=2, padding=1)
        self.conv3 = conv_block(16 * 32, 256, 3, stride=2, padding=1)
        self.conv4 = conv_block(256, 32, 3, padding=1)

        self.scalelayer = ScaleLayer(prep_size)

    def forward(self, x):
        activation = F.leaky_relu
        x = x.view((x.shape[0], self.in_ch, 64, 64))

        if self.arch_prediction:
            x_fc = activation(self.arch_conv(x))
            x_fc = x_fc.view(x_fc.shape[0], -1)
            arch_scores = self.arch_classifier(x_fc)

        x = activation(self.conv1(x))
        x = x.view((x.shape[0], 32 * 4, self.prep_height, 32 * 64))
        x = activation(self.conv2(x))
        x = x.view((x.shape[0], 16 * 32, self.prep_height // 2, self.prep_width // 4))
        x = activation(self.conv3(x))
        x = x.view((x.shape[0], 256, self.prep_height // 4, self.prep_width // 8))
        x = self.conv4(x)
        prep = x.view((x.shape[0], 1, self.prep_height, self.prep_width))

        prep = self.scalelayer(prep)

        if self.arch_prediction:
            return arch_scores, prep
        else:
            return prep

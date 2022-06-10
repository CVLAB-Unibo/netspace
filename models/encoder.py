import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import conv_block


class Encoder(nn.Module):
    def __init__(self, emb_size):
        super(Encoder, self).__init__()
        h = emb_size // 32

        self.conv1 = conv_block(1, h, (1, 3), stride=(1, 2), padding=(0, 1))
        self.conv2 = conv_block(h, h * 2, (1, 3), stride=(1, 2), padding=(0, 1))
        self.conv3 = conv_block(h * 2, h * 4, (3, 1), padding=(1, 0))
        self.conv4 = conv_block(h * 4, h * 32, (3, 1), padding=(1, 0))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = torch.max(x, 3, keepdim=True).values
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = torch.max(x, 2, keepdim=True).values

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

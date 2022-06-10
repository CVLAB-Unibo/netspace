import random

import torch
from models.netfactory import NetFactory
from torch.utils.data import Dataset


class NetsDatasetMissing(Dataset):
    def __init__(self, nets_file, device, eval_func, prep_size, shuffle=False):
        self.prep_size = prep_size
        with open(nets_file) as f:
            lines = [s.strip().split(";") for s in f.readlines()]

        if shuffle:
            random.shuffle(lines)

        self.device = device

        self.nets = []
        nets_A = []
        nets_B = []

        for i, line in enumerate(lines):
            class_name = line[0]
            params = line[1:-1]
            weights_path = line[-1]
            net = NetFactory.get_net(class_name, i, params)
            if len(weights_path) > 0:
                net.load(weights_path)
            net.to(self.device)
            net.eval()
            net.score = eval_func(net)

            if net.class_id == 0:
                nets_A.append(net)
            elif net.class_id == 3:
                nets_B.append(net)

        min_nets_num = min(len(nets_A), len(nets_B))
        max_nets_num = max(len(nets_A), len(nets_B))
        for i in range(min_nets_num):
            self.nets.append([nets_A[i], nets_B[i]])
        for i in range(min_nets_num, max_nets_num, 2):
            if len(nets_A) == max_nets_num:
                self.nets.append([nets_A[i], nets_A[i + 1]])
            else:
                self.nets.append([nets_B[i], nets_B[i + 1]])

    def __len__(self):
        return len(self.nets)

    def __getitem__(self, idx):
        net_1, net_2 = self.nets[idx]
        prep_1 = net_1.get_prep(self.prep_size)
        prep_2 = net_2.get_prep(self.prep_size)
        preps = torch.stack([prep_1, prep_2], dim=0)
        return [net_1, net_2], preps

    @staticmethod
    def collate_fn(batch):
        return batch[0]

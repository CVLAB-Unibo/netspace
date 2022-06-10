import random

import torch
from models.netfactory import NetFactory
from torch.utils.data import Dataset


class NetsDataset(Dataset):
    def __init__(self, nets_file, device, eval_func, prep_size, shuffle=False):
        self.prep_size = prep_size
        with open(nets_file) as f:
            lines = [s.strip().split(";") for s in f.readlines()]

        if shuffle:
            random.shuffle(lines)

        self.device = device

        self.nets = []
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
            self.nets.append(net)

    def __len__(self):
        return len(self.nets)

    def __getitem__(self, idx):
        return self.nets[idx], self.nets[idx].get_prep(self.prep_size)

    @staticmethod
    def collate_fn(batch):
        preps = [element[1] for element in batch]
        prep = torch.stack(preps, dim=0)
        nets = [element[0] for element in batch]
        return [nets, prep]

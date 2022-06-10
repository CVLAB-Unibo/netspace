import os
from typing import Callable, List, Tuple

import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

from data.listdset import ListDataset

CIFAR10_NAME = "cifar10"
TINY_IMAGENET_NAME = "tiny-imagenet-200"


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: List[int], transform: Callable) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        img, label = self.dataset[self.indices[idx]]
        return self.transform(img), label

    def __len__(self) -> int:
        return len(self.indices)


def get_transforms(dataset_name: str, train: bool) -> transforms.Compose:
    t = []

    if train:
        if dataset_name == CIFAR10_NAME:
            t.append(transforms.RandomCrop(32, padding=4))
        elif dataset_name == TINY_IMAGENET_NAME:
            t.append(transforms.RandomCrop(64, padding=8))

        t.append(transforms.RandomHorizontalFlip())

    t.append(transforms.ToTensor())

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t.append(transforms.Normalize(mean, std))

    train_transforms = transforms.Compose(t)

    return train_transforms


class ImagesDataset:
    def __init__(self, dataset_name: str, batch_size: int = 128, data_root: str = "data") -> None:
        dataset_name = dataset_name.lower()
        assert dataset_name in [
            CIFAR10_NAME,
            TINY_IMAGENET_NAME,
        ]
        data_root = os.path.join(data_root, dataset_name)

        self.channels = 3
        self.batch_size = batch_size

        train_transforms = get_transforms(dataset_name, True)
        val_test_transforms = get_transforms(dataset_name, False)

        dset = {
            CIFAR10_NAME: CIFAR10,
            TINY_IMAGENET_NAME: ListDataset,
        }[dataset_name]

        if dataset_name != TINY_IMAGENET_NAME:
            full_trainset = dset(root=data_root, train=True, download=True)
            train_numel = len(full_trainset) // 100 * 80
            train_indices = [i for i in range(train_numel)]
            val_indices = [i for i in range(train_numel, len(full_trainset))]
            self.trainset = Subset(full_trainset, train_indices, train_transforms)
            self.valset = Subset(full_trainset, val_indices, val_test_transforms)

            self.testset = dset(
                root=data_root,
                train=False,
                download=True,
                transform=val_test_transforms,
            )
        else:
            train_file = os.path.join(data_root, "train.txt")
            self.trainset = ListDataset(train_file, train_transforms)
            val_file = os.path.join(data_root, "val.txt")
            self.valset = ListDataset(val_file, val_test_transforms)
            test_file = os.path.join(data_root, "test.txt")
            self.testset = ListDataset(test_file, val_test_transforms)

    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        return self.trainset, self.valset, self.testset

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        trainloader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        valloader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        testloader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        return trainloader, valloader, testloader

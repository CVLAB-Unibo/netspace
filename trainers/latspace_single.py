import os
from typing import Optional

import torch
from data.images import ImagesDataset
from models.basenet import BaseNet
from models.decoder import Decoder
from torch import nn
from torch.nn.functional import kl_div
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from trainers.classification import ClassificationTrainer
from trainers.utils import progress_bar


class LatSpaceTrainer:
    def __init__(
        self,
        dataset_name: str,
        decoder: Decoder,
        emb_size: int,
        device: torch.device,
        logdir: str,
        net: BaseNet,
        teacher_net: nn.Module,
        lr: float,
        embedding: Optional[torch.Tensor] = None,
    ):
        self.decoder = decoder

        dataset = ImagesDataset(dataset_name)
        self.train_loader, self.val_loader, self.test_loader = dataset.get_loaders()

        self.device = device

        self.net = net

        self.eval = ClassificationTrainer.eval_accuracy
        self.summary_writer = SummaryWriter(log_dir=logdir)
        self.best_score = 0.0
        self.best_embedding = None
        self.logdir = logdir

        if embedding is None:
            emb_shape = (1, emb_size, 1, 1)
            self.embedding = torch.randn(
                emb_shape,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )
        else:
            with torch.no_grad():
                pred_prep = self.decoder(embedding)
                val_acc = self.eval(self.net, self.test_loader, pred_prep)
                print("Start val acc:", val_acc)

            self.embedding = torch.tensor(
                embedding.detach().cpu().numpy(),
                dtype=torch.float,
                device=device,
                requires_grad=True,
            )

        self.optimizer = Adam([self.embedding], lr=lr)
        self.teacher_net = teacher_net

    def train(self, epoch_num: int):
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)

        for epoch in range(epoch_num):
            epoch_loss = 0.0
            desc = f"{epoch} / {epoch_num}"
            for images, labels in progress_bar(self.train_loader, desc=desc):
                self.optimizer.zero_grad()

                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    teacher_pred = self.teacher_net(images)

                pred_prep = self.decoder(self.embedding)
                pred = self.net.func_forward(images, pred_prep)

                T = 4.0
                pred = log_softmax(pred / T)
                teacher_pred = softmax(teacher_pred / T)
                loss = kl_div(pred, teacher_pred, reduction="batchmean") * T ** 2

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(self.train_loader)
            self.summary_writer.add_scalar("train/loss", epoch_loss, global_step=epoch)

            with torch.no_grad():
                train_acc = self.eval(self.net, self.train_loader, pred_prep)
                val_acc = self.eval(self.net, self.val_loader, pred_prep)
                self.summary_writer.add_scalar("acc/train", train_acc, global_step=epoch)
                self.summary_writer.add_scalar("acc/val", val_acc, global_step=epoch)

                if val_acc > self.best_score:
                    self.best_score = val_acc
                    self.best_embedding = self.embedding.detach().clone()
                    torch.save(self.best_embedding, os.path.join(self.logdir, "best_embedding.pt"))
                    print("best_score:", self.best_score, "epoch:", epoch)

        self.best_embedding = self.best_embedding.to(self.device)
        pred_prep = self.decoder(self.best_embedding)[0]
        score = self.eval(self.net, self.test_loader, pred_prep)
        print("best embedding score : ", score)

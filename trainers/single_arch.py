from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from trainers.ckptmngr import CheckpointManager
from trainers.classification import ClassificationTrainer
from trainers.utils import progress_bar


class SingleArchTrainer:
    def __init__(self, device, logdir):
        self.device = device
        self.summary_writer = SummaryWriter(log_dir=logdir)
        self.ckptdir = Path(logdir) / "ckpt"

    def train(
        self,
        encoder,
        decoder,
        nets_train_loader,
        nets_val_loader,
        images_train_loader,
        images_val_loader,
        epoch_num,
    ):
        params = list(encoder.parameters())
        params.extend(list(decoder.parameters()))

        optimizer = Adam(params, lr=1e-4)

        elements = [encoder, decoder, optimizer]
        ckptmngr = CheckpointManager(self.ckptdir, 25, lambda x, y: y - x, elements)

        start_epoch = ckptmngr.try_to_load(False)
        print("Training starting from epoch", start_epoch)

        nets_train_iter = iter(nets_train_loader)

        for epoch in range(start_epoch, epoch_num):
            step = 0
            epoch_kl_loss = 0.0

            desc = f"{epoch} / {epoch_num}"
            for images_batch, labels_batch in progress_bar(images_train_loader, desc=desc):
                images_batch = images_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                try:
                    net_batch, prep_batch = next(nets_train_iter)
                except StopIteration:
                    nets_train_iter = iter(nets_train_loader)
                    net_batch, prep_batch = next(nets_train_iter)

                encoder.train()
                decoder.train()

                optimizer.zero_grad()

                embedding = encoder(prep_batch)
                predicted_prep = decoder(embedding)

                kl_loss = self.compute_loss(net_batch, predicted_prep, images_batch, decoder)

                kl_loss.backward()
                optimizer.step()

                step += 1

                epoch_kl_loss += kl_loss.item()

            epoch_kl_loss /= step
            self.summary_writer.add_scalar("train/kl_loss", epoch_kl_loss, epoch)

            if epoch % 20 == 19:
                encoder.eval()
                decoder.eval()

                _ = self.eval(
                    encoder,
                    decoder,
                    nets_train_loader,
                    images_train_loader,
                    epoch,
                    "train_train_accuracy",
                )
                _ = self.eval(
                    encoder,
                    decoder,
                    nets_train_loader,
                    images_val_loader,
                    epoch,
                    "train_val_accuracy",
                )
                validation_error = self.eval(
                    encoder,
                    decoder,
                    nets_val_loader,
                    images_val_loader,
                    epoch,
                    "val_val_accuracy",
                )
                ckptmngr.save_best(epoch, validation_error)

            ckptmngr.save_interval(epoch)

    def compute_loss(self, nets_batch, predicted_prep, images_batch, decoder):
        kl_loss = 0.0
        counter = 0

        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)

        for i in range(len(nets_batch)):
            with torch.no_grad():
                target_pred = nets_batch[i](images_batch)
            net_pred = decoder.out_nets[0].func_forward(images_batch, predicted_prep[i])

            T = 4.0
            target = softmax(target_pred / T)
            pred = log_softmax(net_pred / T)
            kl_loss += F.kl_div(pred, target, reduction="batchmean") * T ** 2

            counter += 1

        kl_loss /= counter

        return kl_loss

    @torch.no_grad()
    def eval(self, encoder, decoder, nets_loader, images_loader, epoch, logdir):
        error = 0
        counter = 0
        evalfn = ClassificationTrainer.eval_accuracy

        for net_batch, prep_batch in nets_loader:
            embedding = encoder(prep_batch)
            predicted_prep = decoder(embedding)

            for i in range(len(net_batch)):
                net = net_batch[i]
                target_score = evalfn(net, images_loader)
                pred_score = evalfn(decoder.out_nets[0], images_loader, predicted_prep[i])
                counter += 1
                error += abs(target_score - pred_score)

                writer_dict = {"target": target_score, "predicted": pred_score}
                self.summary_writer.add_scalars(logdir + "/net" + str(net.id), writer_dict, epoch)

        return error / counter

from itertools import combinations as comb
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from trainers.ckptmngr import CheckpointManager
from trainers.classification import ClassificationTrainer
from trainers.utils import progress_bar


class MultiArchTrainer:
    def __init__(self, device, logdir):
        self.device = device
        self.summary_writer = SummaryWriter(log_dir=logdir)
        self.ckptdir = Path(logdir) / "ckpt"
        self.eval_fn = ClassificationTrainer.eval_accuracy

    def train(
        self,
        encoder,
        decoder,
        nets_train_loader,
        nets_val_loader,
        images_train_loader,
        images_val_loader,
        epoch_num,
        num_archs,
        teacher_net,
    ):
        params = list(encoder.parameters())
        params.extend(list(decoder.parameters()))
        optimizer = Adam(params, lr=1e-4)

        elements = [encoder, decoder, optimizer]
        ckptmngr = CheckpointManager(self.ckptdir, 25, lambda x, y: x - y, elements)

        start_epoch = ckptmngr.try_to_load(False)
        print("Training starting from epoch", start_epoch)

        nets_train_iter = iter(nets_train_loader)

        for epoch in range(start_epoch, epoch_num):
            step = 0
            epoch_kd_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_arch_loss = 0.0
            epoch_interp_loss = 0.0
            epoch_interp_arch_loss = 0.0
            epoch_interp_kd_loss = 0.0
            epoch_total_loss = 0.0

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

                embeddings = encoder(prep_batch)

                # we want to activate interpolation loss only on boundaries
                interp_distance = max(1, num_archs - 2)

                interp_arch_loss, interp_kl_loss, interp_ce_loss = self.interpolation_loss(
                    decoder,
                    embeddings,
                    net_batch,
                    images_batch,
                    labels_batch,
                    teacher_net,
                    interp_distance,
                )

                w = 0.9
                interp_kd_loss = w * interp_kl_loss + (1 - w) * interp_ce_loss
                interp_loss = interp_arch_loss + interp_kd_loss

                predicted_archs, predicted_preps = decoder(embeddings)

                arch_loss, kl_loss, ce_loss = self.compute_losses(
                    net_batch,
                    predicted_preps,
                    predicted_archs,
                    images_batch,
                    labels_batch,
                    decoder,
                    teacher_net,
                )

                w = 0.9
                kd_loss = w * kl_loss + (1 - w) * ce_loss

                optimizer.zero_grad()

                total_loss = arch_loss + kd_loss + interp_loss
                total_loss.backward()
                optimizer.step()
                step += 1

                epoch_ce_loss += ce_loss.item()
                epoch_kd_loss += kd_loss.item()
                epoch_interp_loss += interp_loss.item()
                epoch_interp_arch_loss += interp_arch_loss.item()
                epoch_interp_kd_loss += interp_kd_loss.item()
                epoch_arch_loss += arch_loss.item()
                epoch_total_loss += total_loss.item()

            epoch_ce_loss /= step
            epoch_kd_loss /= step
            epoch_interp_loss /= step
            epoch_interp_arch_loss /= step
            epoch_interp_kd_loss /= step
            epoch_arch_loss /= step
            epoch_total_loss /= step

            self.summary_writer.add_scalar("train/ce_loss", epoch_ce_loss, epoch)
            self.summary_writer.add_scalar("train/kd_loss", epoch_kd_loss, epoch)
            self.summary_writer.add_scalar("train/arch_loss", epoch_arch_loss, epoch)
            self.summary_writer.add_scalar("train/interp_loss", epoch_interp_loss, epoch)
            self.summary_writer.add_scalar("train/interp_arch_loss", epoch_interp_arch_loss, epoch)
            self.summary_writer.add_scalar("train/interp_kd_loss", epoch_interp_kd_loss, epoch)
            self.summary_writer.add_scalar("train/total_loss", epoch_total_loss, epoch)

            if epoch % 25 == 24:
                encoder.eval()
                decoder.eval()

                _ = self.eval(
                    epoch,
                    encoder,
                    decoder,
                    nets_train_loader,
                    images_train_loader,
                    "train_train_accuracy",
                    num_archs,
                )
                _ = self.eval(
                    epoch,
                    encoder,
                    decoder,
                    nets_train_loader,
                    images_val_loader,
                    "train_val_accuracy",
                    num_archs,
                )
                validation_error = self.eval(
                    epoch,
                    encoder,
                    decoder,
                    nets_val_loader,
                    images_val_loader,
                    "val_val_accuracy",
                    num_archs,
                )
                ckptmngr.save_best(epoch, validation_error)
            ckptmngr.save_interval(epoch)

    def interpolation_loss(
        self,
        decoder,
        embeddings,
        nets_batch,
        images_batch,
        labels_batch,
        teacher_net,
        working_distance=1,
        T=4,
    ):
        samples = []
        archs = []
        softmaxes = []
        softmax = nn.Softmax(dim=1)
        log_softmax = nn.LogSoftmax(dim=1)

        for couple in comb([i for i in range(embeddings.shape[0])], 2):
            i, j = couple
            if nets_batch[i].class_id > nets_batch[j].class_id:
                j, i = couple
            class_min = nets_batch[i].class_id
            class_max = nets_batch[j].class_id
            embedding_min = embeddings[i].view(-1)
            embedding_max = embeddings[j].view(-1)
            teacher_softmax = softmax(teacher_net(images_batch) / T)
            distance = class_max - class_min
            if distance > working_distance:
                for step in range(1, distance):
                    inter_factor = (1 / distance) * step
                    archs.append(class_min + step)
                    sample = (1 - inter_factor) * embedding_min + (inter_factor) * embedding_max
                    samples.append(sample)
                    target_softmax = teacher_softmax
                    softmaxes.append(target_softmax)

        arch_loss = torch.tensor([0.0]).to(self.device)
        kl_loss = torch.tensor([0.0]).to(self.device)
        ce_loss = torch.tensor([0.0]).to(self.device)
        if len(samples) > 0:
            samples_batch = torch.stack(samples, dim=0).unsqueeze(2).unsqueeze(3)
            pred_arch, pred_prep = decoder(samples_batch)
            arch_loss = F.cross_entropy(
                pred_arch, torch.tensor(archs).to(self.device), reduction="sum"
            )

            for i in range(len(softmaxes)):
                arch = torch.argmax(pred_arch[i], dim=0).item()
                net = decoder.out_nets[arch]
                pred = net.func_forward(images_batch, pred_prep[i])
                pred_logs = log_softmax(pred / T)
                kl = F.kl_div(pred_logs, softmaxes[i], reduction="batchmean") * T ** 2
                kl_loss += kl
                ce = nn.functional.cross_entropy(pred, labels_batch)
                ce_loss += ce

        return arch_loss, kl_loss, ce_loss

    def compute_losses(
        self,
        nets_batch,
        predicted_preps,
        predicted_archs,
        images_batch,
        labels_batch,
        decoder,
        teacher_net,
        T=4,
    ):
        ce_loss = 0.0
        kl_loss = 0.0
        arch_loss = 0.0

        for i in range(len(nets_batch)):
            with torch.no_grad():
                target_pred = teacher_net(images_batch)
            predicted_arch = predicted_archs[i].unsqueeze(0)
            target_arch = torch.tensor([nets_batch[i].class_id]).to(self.device)
            arch_loss += F.cross_entropy(predicted_arch, target_arch)

            arch = torch.argmax(predicted_archs[i], dim=0).item()
            predicted_net = decoder.out_nets[arch]
            net_pred = predicted_net.func_forward(images_batch, predicted_preps[i])

            ce = nn.functional.cross_entropy(net_pred, labels_batch)
            ce_loss += ce

            softmax = nn.Softmax(dim=1)
            log_softmax = nn.LogSoftmax(dim=1)

            target = softmax(target_pred / T)
            pred = log_softmax(net_pred / T)
            kl = F.kl_div(pred, target, reduction="batchmean") * T ** 2
            kl_loss += kl

        return arch_loss, kl_loss, ce_loss

    @torch.no_grad()
    def eval(self, epoch, encoder, decoder, nets_loader, images_loader, logdir, num_archs):
        counter = 0
        score = 0
        inter_perfs = {}
        inter_class_examples_num = {}

        for i in range(num_archs):
            inter_perfs[i] = 0
            inter_class_examples_num[i] = 0

        min_class_id = 0
        max_class_id = num_archs - 1
        min_embeddings = []
        max_embeddings = []

        for net_batch, prep_batch in nets_loader:
            embedding = encoder(prep_batch)
            predicted_archs, predicted_preps = decoder(embedding)

            for i in range(len(net_batch)):
                net = net_batch[i]
                if net.class_id == min_class_id:
                    min_embeddings.append(embedding[i])
                elif net.class_id == max_class_id:
                    max_embeddings.append(embedding[i])
                predicted_arch = predicted_archs[i].unsqueeze(0)
                target_arch = torch.tensor([net.class_id]).to(self.device)
                arch_loss = F.cross_entropy(predicted_arch, target_arch)

                target_score = self.eval_fn(net, images_loader)
                arch = torch.argmax(predicted_archs[i], dim=0).item()
                predicted_net = decoder.out_nets[arch]
                pred_score = self.eval_fn(predicted_net, images_loader, predicted_preps[i])
                writer_dict = {"target": target_score, "predicted": pred_score}

                counter += 1
                score += pred_score

                self.summary_writer.add_scalars(logdir + "/net" + str(net.id), writer_dict, epoch)
                self.summary_writer.add_scalar(logdir + "/net_arch" + str(net.id), arch_loss, epoch)

        for i in range(len(min_embeddings)):
            distance = max_class_id - min_class_id
            for step in range(1, distance):
                inter_factor = (1 / distance) * step
                sample = (1 - inter_factor) * min_embeddings[i] + (inter_factor) * max_embeddings[i]
                sample = sample.unsqueeze(0)
                sample_arch, sample_prep = decoder(sample)
                arch = torch.argmax(sample_arch[0], dim=0).item()
                predicted_net = decoder.out_nets[arch]
                pred_score = self.eval_fn(predicted_net, images_loader, sample_prep[0])
                score += pred_score
                counter += 1
                inter_perfs[arch] += pred_score
                inter_class_examples_num[arch] += 1

        for i in range(num_archs):
            if inter_class_examples_num[i] != 0:
                inter_perfs[i] /= inter_class_examples_num[i]
                writer_dict = {"predicted": inter_perfs[i]}
                self.summary_writer.add_scalars(logdir + "/interp_net" + str(i), writer_dict, epoch)

        return score / counter

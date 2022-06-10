import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from trainers.utils import adjust_learning_rate, progress_bar


class ClassificationTrainer:
    def __init__(self, device, log_dir, ckpt_path, save_mode, show_progress=False):
        self.device = device
        self.ckpt_path = ckpt_path
        self.save_mode = save_mode
        self.best_val_acc = 0
        self.show_progress = show_progress

        self.summary_writer = None
        if log_dir is not None:
            self.summary_writer = SummaryWriter(log_dir=log_dir)

    def train(self, net, train_loader, val_loader, epoch_num, optimizer, lr_value, lr_decay_epochs):
        ce = nn.CrossEntropyLoss()

        for epoch in range(epoch_num):
            net.train()

            if self.show_progress:
                iterable = progress_bar(train_loader, f"Epoch {epoch}")
            else:
                iterable = train_loader

            for data in iterable:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = net(inputs)
                loss = ce(outputs, labels)
                loss.backward()
                optimizer.step()

            if self.summary_writer and epoch % 20 == 19:
                net.eval()
                validation_acc = self.eval_accuracy(net, val_loader)
                training_acc = self.eval_accuracy(net, train_loader)
                self.summary_writer.add_scalar("val_acc", validation_acc, epoch)
                self.summary_writer.add_scalar("train_acc", training_acc, epoch)

            if self.save_mode == "best" and epoch >= 200 and epoch % 10 == 9:
                net.eval()
                validation_acc = self.eval_accuracy(net, val_loader)
                if validation_acc > self.best_val_acc:
                    self.best_val_acc = validation_acc
                    net.save(self.ckpt_path)

            adjust_learning_rate(epoch, lr_decay_epochs, optimizer, lr_value)

        if self.save_mode == "final":
            net.save(self.ckpt_path)

    @staticmethod
    def eval_accuracy(net, images_loader, net_prep=None, device="cuda:0"):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in images_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                if net_prep is not None:
                    outputs = net.func_forward(images, net_prep)
                else:
                    outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        return accuracy

from pathlib import Path
from random import sample

import torch
import torch.nn.functional as F
from models.mlp import MLP
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from trainers.ckptmngr import CheckpointManager
from trainers.utils import progress_bar, sdf_to_mesh


class MLPTrainer:
    def __init__(self, device, logdir):
        self.device = device
        self.summary_writer = SummaryWriter(log_dir=logdir)
        self.ckptdir = Path(logdir) / "ckpt"

    def train(self, encoder, decoder, mlps_loader, num_epochs):
        params = list(encoder.parameters())
        params.extend(list(decoder.parameters()))

        optimizer = Adam(params, lr=1e-4)

        elements = [encoder, decoder, optimizer]
        ckptmngr = CheckpointManager(self.ckptdir, 25, lambda x, y: x - y, elements)

        start_epoch = ckptmngr.try_to_load(False)
        print("Training starting from epoch", start_epoch)

        for epoch in range(start_epoch, num_epochs):
            encoder.train()
            decoder.train()

            step = 0
            epoch_loss = 0.0

            desc = f"{epoch} / {num_epochs}"
            for preps, coords in progress_bar(mlps_loader, desc=desc):
                preps = preps.to(self.device)
                coords = coords.to(self.device)

                embeddings = encoder(preps)
                predicted_preps = decoder(embeddings)

                loss = self.compute_loss(coords, preps, predicted_preps)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1

                epoch_loss += loss.item()

            epoch_loss /= step
            self.summary_writer.add_scalar("train/loss", epoch_loss, epoch)

            if epoch % 1000 == 999:
                encoder.eval()
                decoder.eval()

                acc = self.eval(encoder, decoder, mlps_loader, "train", epoch)

                ckptmngr.save_best(epoch, acc)

                self.plot(encoder, decoder, mlps_loader, "train", epoch)

            ckptmngr.save_interval(epoch)

    @staticmethod
    def mlp_batched_forward(coordinates, preps):
        params = [MLP.params_from_prep(prep) for prep in preps]
        batched_params = []
        for i in range(len(params[0])):
            batched_params.append(torch.stack([p[i] for p in params], dim=0))

        num_layers = len(batched_params) // 2

        f = coordinates

        for i in range(num_layers):
            weights = batched_params[i * 2]
            biases = batched_params[i * 2 + 1]

            f = f @ weights.permute(0, 2, 1) + biases.unsqueeze(1)

            if i < num_layers - 1:
                f = torch.sin(30 * f)

        return f

    def compute_loss(self, coords, input_preps, predicted_preps):
        with torch.no_grad():
            y = MLPTrainer.mlp_batched_forward(coords, input_preps)
        pred = MLPTrainer.mlp_batched_forward(coords, predicted_preps)

        loss = F.mse_loss(pred, y)
        return loss

    @torch.no_grad()
    def eval(self, encoder, decoder, mlps_loader, logtag, epoch):
        correct = 0
        counter = 0

        for preps, coords in mlps_loader:
            preps = preps.to(self.device)
            coords = coords.to(self.device)

            embeddings = encoder(preps)
            predicted_preps = decoder(embeddings)

            y = MLPTrainer.mlp_batched_forward(coords, preps)
            pred = MLPTrainer.mlp_batched_forward(coords, predicted_preps)
            correct += torch.count_nonzero(torch.isclose(pred, y, rtol=0.0, atol=1e-2))

            counter += coords.shape[0] * coords.shape[1]

        acc = float(correct / counter)

        self.summary_writer.add_scalar(logtag + "/acc", acc, epoch)

        return acc

    @torch.no_grad()
    def plot(self, encoder, decoder, mlps_loader, logtag, epoch):
        dataset = mlps_loader.dataset
        indices = sample(range(len(dataset)), 4)

        for n, idx in enumerate(indices):
            prep, _ = dataset[idx]
            prep = prep.to(self.device)

            embedding = encoder(prep.unsqueeze(0))
            predicted_prep = decoder(embedding).squeeze(0)

            mlp = MLP(256, 1).to(self.device)
            sd = mlp.state_dict()

            try:
                params = MLP.params_from_prep(prep)
                sd = {k: params[i] for i, k in enumerate(sd)}
                mlp.load_state_dict(sd)
                gt_vertices, gt_faces = sdf_to_mesh(mlp, grid_size=128)
                self.summary_writer.add_mesh(
                    logtag + f"/gt_{n}",
                    vertices=torch.tensor(gt_vertices.copy()).unsqueeze(0),
                    faces=torch.tensor(gt_faces.copy()).unsqueeze(0),
                    global_step=epoch,
                )

                params = MLP.params_from_prep(predicted_prep)
                sd = {k: params[i] for i, k in enumerate(sd)}
                mlp.load_state_dict(sd)
                pred_vertices, pred_faces = sdf_to_mesh(mlp, grid_size=128)
                self.summary_writer.add_mesh(
                    logtag + f"/pred_{n}",
                    vertices=torch.tensor(pred_vertices.copy()).unsqueeze(0),
                    faces=torch.tensor(pred_faces.copy()).unsqueeze(0),
                    global_step=epoch,
                )
            except:
                pass

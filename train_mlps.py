from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.mlpdset import MLPDataset
from models.decoder import Decoder
from models.encoder import Encoder
from trainers.mlps import MLPTrainer

device = torch.device("cuda")

emb_size = 4096
prep_shape = (8, 10_000)
mlp_batch_size = 4
num_epochs = 1_000_000

logdir = f"/path/to/log/dir"

Path(logdir).mkdir(parents=True, exist_ok=True)

dataset_root = Path("/path/to/sdf/dataset")
train_range = (0, 1000)
num_coords = 50_000

mlps_dataset = MLPDataset(dataset_root, train_range, prep_shape, num_coords)
mlps_dataloader = DataLoader(
    mlps_dataset,
    batch_size=mlp_batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

encoder = Encoder(emb_size).to(device)
decoder = Decoder([], emb_size, prep_shape, arch_prediction=False).to(device)

trainer = MLPTrainer(device, logdir)
trainer.train(encoder, decoder, mlps_dataloader, num_epochs)

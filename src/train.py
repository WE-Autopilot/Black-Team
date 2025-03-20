import torch as pt
from torch.optim import AdamW
from sal import SAL
from dataset import Stage0Dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import h5py as hp
import numpy as np


batch_size = 64
epochs = 10

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

sal = SAL(num_points=16).to(device)
state_dict = pt.load("backup.ckpt", map_location=device)
sal.load_state_dict(state_dict)
optimizer = AdamW(sal.parameters(), lr=1e-4)
dataset = Stage0Dataset("dataset.h5", np.arange(16))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses = []
pbar = tqdm(range(epochs), desc="training...", unit="epoch")
for i in pbar:
    total_loss = 0
    pbar1 = tqdm(dataloader, desc=f"E{i}", unit="batch")
    for images, target in pbar1:
        images = images.to(device)
        target = target.to(device)
        pos = pt.zeros(len(images), 2).to(device)

        dist, value = sal(images, pos)
        log_probs = dist.log_prob(target)

        loss = -log_probs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.mean().item()
        pbar1.set_postfix(L=f"{loss.mean().item():.4f}")

    avg_loss = total_loss / len(dataloader)
    pbar.set_postfix(L=f"{avg_loss:.4f}")
    losses.append(avg_loss)

    pt.save(sal.state_dict(), 'backup.ckpt')

with hp.File("logs.h5", "w") as file:
    file.create_dataset("loss", data=losses)

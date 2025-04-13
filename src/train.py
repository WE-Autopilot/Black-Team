import torch as pt
from torch.optim import AdamW
from sal import SAL
from dataset import Stage0Dataset
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import h5py as hp
import numpy as np


batch_size = 2 ** 10
epochs = 3
velocity = 1

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

sal = SAL(1080).to(device)
try:
    1/0
    state_dict = pt.load("backup.ckpt", map_location=device)
    sal.load_state_dict(state_dict)
    print(f"Loading backup model on {device}")
except:
    print(f"Loading blank model on {device}")
optimizer = AdamW(sal.parameters(), lr=1e-4)
dataset = Stage0Dataset("dataset.h5")
full_dataset_size = len(dataset)
subset_size = int(0.1 * full_dataset_size)
indices = pt.randperm(full_dataset_size).tolist()
subset_indices = indices[:subset_size]
subset = Subset(dataset, subset_indices)
dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

losses = []
pbar = tqdm(range(epochs), desc="training...", unit="epoch")
for i in pbar:
    total_loss = 0
    pbar1 = tqdm(dataloader, desc=f"E{i}", unit="batch")
    for scans, target in pbar1:
        scans = scans.to(device)
        target = target.unsqueeze(1)
        vel = pt.full_like(target, velocity)
        target = pt.cat((target, vel), dim=1)
        target = target.to(device)

        dist, value = sal(scans)
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

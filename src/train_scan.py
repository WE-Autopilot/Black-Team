import torch as pt
from torch.optim import AdamW
from scansal import ScanDataset, ScanSAL
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import h5py as hp
import numpy as np


batch_size = 2**10
epochs = 10

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

sal = ScanSAL(num_points=16).to(device)
try:
    1/0
    state_dict = pt.load("backup.ckpt", map_location=device)
    sal.load_state_dict(state_dict)
    print(f"Loaded backup model on {device}")
except:
    print(f"Using blank model on {device}")
optimizer = AdamW(sal.parameters(), lr=1e-4)
dataset = ScanDataset("dataset.h5", np.arange(16))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses = []
pbar = tqdm(range(epochs), desc="training...", unit="epoch")
for i in pbar:
    total_loss = 0
    pbar1 = tqdm(dataloader, desc=f"E{i}", unit="batch")
    for images, target in pbar1:
        images = images.to(device)
        target = target.to(device)

        dist = sal(images)
        log_probs = dist.log_prob(target)

        loss = -log_probs.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.mean().item()
        pbar1.set_postfix(L=f"{loss.mean().item():.4f}")
        losses.append(loss.mean().item())

    avg_loss = total_loss / len(dataloader)
    pbar.set_postfix(L=f"{avg_loss:.4f}")

    pt.save(sal.state_dict(), 'backup.ckpt')

    with hp.File("logs.h5", "w") as file:
        file.create_dataset("loss", data=losses)

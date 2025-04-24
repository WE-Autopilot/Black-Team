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
state_dict = pt.load("model.ckpt", map_location=device)
sal.load_state_dict(state_dict)
dataset = Stage0Dataset("val_dataset.h5", np.arange(16))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

pbar = tqdm(dataloader, desc=f"E{i}", unit="batch")
all_losses = pt.tensor([])
for images, target in pbar:
    images = images.to(device)
    target = target.to(device)
    pos = pt.zeros(len(images), 2).to(device)

    dist, value = sal(images, pos)
    log_probs = dist.log_prob(target)

    losses = -log_probs.mean(dim=-1)
    all_losses = pt.cat((all_losses, losses))

    total_loss += losses.mean().item()
    pbar.set_postfix(L=f"{losses.mean().item():.4f}")



print

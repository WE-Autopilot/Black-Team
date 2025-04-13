import torch as pt
from torch.optim import AdamW
from sal import SAL
from dataset import Stage0Dataset, show_path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import h5py as hp
import numpy as np


batch_size = 64
epochs = 10

#device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
device = pt.device("cpu")

print("Loading model...")
sal = SAL(num_points=16).to(device)
state_dict = pt.load("model.ckpt", map_location=device)
sal.load_state_dict(state_dict)
sal.eval()
print("Loading dataset...")
dataset = Stage0Dataset("dataset.h5", np.arange(16))

pbar = tqdm(range(len(dataset)), desc="test...", unit="sample")
for i in pbar:
    image, target = dataset[i]
    image = image.unsqueeze(0).to(device)
    target = target.to(device)
    pos = pt.zeros(1, 2).to(device)

    dist, value = sal(image, pos)
    paths = [dist.sample() for _ in range(100)]

    show_path(image[0], paths)

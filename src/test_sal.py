import torch as pt
from torch.optim import AdamW
from sal import SAL
from dataset import Stage0Dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
import h5py as hp
import numpy as np
from weap_util.lidar import lidar_to_bitmap


polar_vec = lambda angle, mag: np.array([mag * np.cos(angle), mag * np.sin(angle)])


interval = 4

#device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
device = pt.device("cpu")

print("Loading model...")
sal = SAL().to(device)
state_dict = pt.load("model.ckpt", map_location=device)
sal.load_state_dict(state_dict)
sal.eval()
print("Loading dataset...")
dataset = Stage0Dataset("dataset.h5")

fig, ax = plt.subplots()
ims = []
pbar = tqdm(range(4096), desc="test...", unit="sample")
for i in pbar:
    i *= interval
    i = i // 256 * 2560 + i % 256
    scan, target = dataset[i]
    scan = scan.unsqueeze(0).to(device)
    target = target.to(device)

    dist, value = sal(scan)
    [[steer, speed]] = dist.mean.detach()
    
    image = lidar_to_bitmap(scan=scan[0], channels=1, fov=2 * np.pi, draw_mode="FILL", bg_color="black", draw_center=False)

    img = ax.imshow(image, origin="lower", animated=True)
    x, y = 128, 128
    dx, dy = polar_vec(np.pi / 2 + steer, speed) * 10
    dx1, dy1 = polar_vec(np.pi / 2 + target, 100)
    point = ax.scatter(x, y, animated=True)
    arrow = ax.arrow(x, y, dx, dy, head_width=4, head_length=4, fc='black', ec='black', animated=True)
    target_arrow = ax.arrow(x, y, dx1, dy1, head_width=5, head_length=5, fc='red', ec='red', animated=True)

    ims.append([img, point, arrow, target_arrow])

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
ani.save('test.mp4', writer='ffmpeg', fps=60)

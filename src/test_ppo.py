import torch as pt
from torch.optim import AdamW
from sal import SAL
from ppo_utils import ppo_update
from toy_dataset import PixelIndexer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def plot_images_with_points(images, points, path):
    """
    Plots a batch of images with their corresponding points on a single figure.

    Args:
        images (torch.Tensor): Tensor of shape (64, 32, 32) representing the images.
        points (torch.Tensor): Tensor of shape (64, 2) representing the points.
    """

    num_images = images.shape[0]
    rows = 8  # Adjust as needed (e.g., 8 rows x 8 columns = 64 images)
    cols = 8

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))  # Adjust figsize as needed

    for i in range(num_images):
        row = i // cols
        col = i % cols

        ax = axes[row, col]
        ax.imshow(images[i].cpu().numpy(), cmap='gray')
        ax.scatter(points[i, 0].cpu().numpy(), points[i, 1].cpu().numpy(), c='red', s=10) #plot the points

        ax.axis('off')  # Turn off axis labels and ticks for cleaner display

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.savefig(path)
    plt.close(fig) #close figure to prevent memory leak.


batch_size = 64
epochs = 1000

device = pt.device("cuda" if pt.cuda.is_available() else "cpu") #determine device.

sal = SAL(num_points=1).to(device) #move model to device.
optimizer = AdamW(sal.parameters(), lr=1e-5)
dataset = PixelIndexer(32, 32)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if not os.path.exists("graphs"): #create graphs directory if it does not exist.
    os.makedirs("graphs")

losses = []
pbar = tqdm(range(epochs), desc="training...", unit="epoch")
for i in pbar:
    total_loss = 0
    for images, target in dataloader:
        images = images.to(device) #move data to device.
        target = target.to(device)
        pos = pt.zeros(len(images), 2).to(device) #move pos to device.

        dist, value = sal(images, pos)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        loss = pt.linalg.norm(actions - target, dim=-1)
        advantages = loss - value

        ppo_update(sal, optimizer, images, pos, actions, log_probs.detach(), loss.detach(), advantages.detach(), entropy_coef=0.1, mini_batch_size=batch_size, epochs=8)

        total_loss += loss.mean().item()

    avg_loss = total_loss / len(dataloader)
    pbar.set_postfix(L=f"{avg_loss:.2f}")
    losses.append(avg_loss)

    if i % 25 == 0:
        plot_images_with_points(images[:, 0].detach().cpu(), actions.detach().cpu(), f"graphs/{i}.png") #move data back to cpu for plotting.

plt.plot(losses)
plt.savefig("graphs/losses.png")

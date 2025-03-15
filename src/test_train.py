import torch as pt
from torch.optim import AdamW
from sal import SAL
from ppo_utils import ppo_update
from toy_dataset import PixelIndexer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt


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


batch_size = 64
epochs = 100

sal = SAL(num_points=1)
optimizer = AdamW(sal.parameters(), lr=1e-4)
dataset = PixelIndexer(32, 32)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

pbar = tqdm(range(epochs), desc="training...", unit="epoch")
for i in pbar:
    for images, target in dataloader:
        pos = pt.zeros(len(images), 2)

        dist, value = sal(images, pos)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        loss = pt.linalg.norm(actions - target, dim=-1)
        advantages = loss - value

        ppo_update(sal, optimizer, images, pos, actions, log_probs.detach(), loss.detach(), advantages.detach(), mini_batch_size=batch_size)

        pbar.set_postfix(L=f"{loss.mean().item():.2f}")

    plot_images_with_points(images[:, 0], actions, f"graphs/{i}.png")

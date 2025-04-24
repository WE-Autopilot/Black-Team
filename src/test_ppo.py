import torch as pt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from test_ppo_utils import ToyDataset, Model
from ppo_utils import ppo_update
import matplotlib.pyplot as plt

len = 128
epochs = 1000

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

dataset = ToyDataset(len)
model = Model(len).to(device)
dataloader = DataLoader(dataset, batch_size=len)
optimizer = AdamW(model.parameters(), lr=1e-4)

mean_rewards = pt.zeros(epochs)
mean_entropy = pt.zeros(epochs)
pbar = tqdm(range(epochs))
for epoch in pbar:
    for states, targets in dataloader:
        states = states.to(device)
        targets = targets.to(device)

        dist, value = model(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        rewards = -pt.abs(actions - targets) // 10
        advantages = rewards - value

        ppo_update(model, optimizer, states, actions, log_probs, rewards, advantages, device=device, entropy_coef=1e-6)

        mean_rewards[epoch] = rewards.mean()
        mean_entropy[epoch] = dist.entropy().mean().detach()

    pbar.set_postfix(L=f"{mean_rewards[epoch]:.4f}")

plt.plot(mean_rewards)
plt.plot(mean_entropy)
plt.show()

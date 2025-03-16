import torch as pt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss
from tqdm import tqdm


# PPO update function remains the same.
def ppo_update(model, optimizer, images, positions, actions, log_probs_old, losses, advantages, clip_param=0.2, value_loss_coef=0.5, entropy_coef=0.01, epochs=4, mini_batch_size=16, shuffle=True):

    log_probs_old = log_probs_old.detach()
    losses = losses.detach()
    advantages = advantages.detach()

    dataset = TensorDataset(images, positions, actions, log_probs_old, losses, advantages)
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=shuffle)

    #pbar = tqdm(range(epochs), desc="training...", unit="epoch")
    for _ in range(epochs):
        for batch_images, batch_positions, batch_actions, batch_log_probs_old, batch_losses, batch_advantages in dataloader:
            dist, values = model(batch_images, batch_positions)
            
            batch_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            ratio = pt.exp(batch_log_probs - batch_log_probs_old).mean(dim=-1)
            surr1 = ratio * batch_advantages
            surr2 = pt.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
            policy_loss = pt.min(surr1, surr2).mean()
            
            value_loss = mse_loss(batch_losses, values)
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            #pbar.set_postfix(L=f"{loss.item():.4f}")
            #print(f"\n\n{entropy.item():.2f} {policy_loss.item():.2f} {value_loss.item():.2f} {loss.item():.2f}\n\n")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


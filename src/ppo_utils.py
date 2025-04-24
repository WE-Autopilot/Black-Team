import torch as pt 
import numpy as np 

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, clip_param = 0.2,
     policy_coef = 0.5, value_loss_coef = 0.5, entropy_coef = 0.01, epochs = 4, mini_batch_size = 16, device = 'cpu') :
    """Performs the ppo update

    Args:
        model (pt model): PolicyValueNet model
        optimizer (optim.Optimizer): The optimizer to use for training
        states (np.array): Batch of iamges observations
        actions (np.array): Batch of actions taken
        old_log_probs (np.array): log propabilites of actions
        returns (np.array): Batch of calculated returns 
        advantages (np.array): Batch of calculated advantages (V(s') - V(s))
        clip_param (float, optional): PPO clip parameter (clip ratio). Defaults to 0.2.
        policy_coef (float, optional): Coefficient for policy loss. Defaults to 0.5.
        value_loss_coef (float, optional): Coefficient for value loss. Defaults to 1.0 - policy_coef
        entropy_coef (float, optional): Coefficient for entropy. Defaults to 0.01.
        epochs (int, optional): Number of optimization epochs. Defaults to 4.
        mini_batch_size (int, optional): Size of mini-batches. Defaults to 16.
        device (str, optional): CPU or GPU device. Defaults to 'cpu'.
    """

    # detach
    old_log_probs = old_log_probs.detach() 
    returns = returns.detach() 
    advantages = advantages.detach() 
    
    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False) if advantages.numel() > 1 else pt.tensor(1.0, device=advantages.device)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    
    batch_size = states.shape[0]
    for _ in range(epochs) :
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        
        for start in range(0, batch_size, mini_batch_size) :
            mb_indices = indices[start:start + mini_batch_size]
            mb_states = states[mb_indices].clone()
            mb_actions = actions[mb_indices]
            mb_log_probs_old = old_log_probs[mb_indices]
            mb_returns = returns[mb_indices]
            mb_advantages = advantages[mb_indices]

            # Calculate Loss
            new_dist, value = model(mb_states)
            # action_dist = Categorical(logits=action_logits)
            log_probs = new_dist.log_prob(mb_actions).sum(dim=-1)

            # Calculate the policy ratio (pi_new / pi_old)
            ratio = pt.exp(log_probs - mb_log_probs_old)

            # Calculate the clipped surrogate objective (Policy Loss)
            surr1 = ratio * mb_advantages 
            surr2 = pt.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
            policy_loss = -pt.min(surr1, surr2).mean()


            # calculate loss (MSE) 
            value_loss = pt.nn.functional.mse_loss(value.squeeze(), mb_returns.squeeze())
            entropy = new_dist.entropy().sum(dim=-1).mean()

            value_loss_coef = 1.0 - policy_coef - entropy_coef
            
            # Calculate the total loss
            loss = policy_coef * policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            
            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

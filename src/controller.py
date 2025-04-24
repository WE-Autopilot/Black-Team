import numpy as np
from weap_util.abstract_controller import AbstractController
from sal import SAL
import torch as pt
import torch.optim as optim
from ppo_utils import ppo_update
from reward_utils import get_progress

def addNoise(obs):
    """
    Adds noise to the observation data.

    We can add the rolling stuff here @Ian
    """
    return obs

class Controller(AbstractController):
        
    def __init__(self, path="model.ckpt") -> None:
        super().__init__()
        self.sal = SAL()
        self.sal.load(path)

    def startup(self, wpts):

        self.values = []
        self.scans = []
        self.actions = []
        self.log_probs = []
        self.progress = []
        self.last_pos = None

        self.waypoints = wpts[:, :2]

        print(self.sal.fc_layers[0].bias)

        pass

    def compute(self, obs):
        """
        Computes control commands and returns the current set of global waypoints.
        It checks if the vehicle is near the last few waypoints and loads the next batch if needed.
        """
        current_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        if self.last_pos is not None:
            self.progress.append(get_progress(self.waypoints, self.last_pos, current_pos))
        
        self.last_pos = current_pos

        scans = pt.tensor(obs["scans"][0], dtype=pt.float)[None, ...]
        dist, val = self.sal(scans)
        self.scans.append(scans)
    
        action = dist.sample()

        [[steer, speed]] = action.tolist()

        self.values.append(val.item())
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))

        # LOGGING
        # print(f"Speed: {speed:.2f}")
        # print(f"Steer: {steer:.2f}")
        # print(f"Mean: {dist.mean[0, 0]}\n")
        # print(f"Standard Deviation: {dist.stddev[0, 0]}")

        return speed*7, steer 

    def shutdown(self):
        pass

    def train_update(self, obs):
        """
        3 Things TODO

        1. Convert the lists above into tensors using pt.cat
        2. calculate the reward which is the progress, clone that value into a tensor which is the same length as the value list
            a. Subtract ten points if it crashes
        3. calculate the advantage which is the reward - value
        4. calculate ppo update with all the tensors from above

        NOTE: Make sure we don't change the dimensions at all, the lists should be its own dimension within the tensor.
        """
        
        crashed = obs["collisions"]

        current_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        self.progress.append(get_progress(self.waypoints, self.last_pos, current_pos))

        # Convert lists to tensors using pt.cat
        scans_tensor = pt.cat(self.scans, dim=0)          # [T, ...] observations
        actions_tensor = pt.cat(self.actions, dim=0)        # [T, ...] actions taken
        log_probs_tensor = pt.cat(self.log_probs, dim=0).sum(dim=-1)    # [T, ...] logged probabilities
        progress_tensor = pt.tensor(self.progress, dtype=pt.float32)  # [T] progress made
        # Unsqueeze to maintain the extra dimension per timestep.
        values_tensor = pt.tensor(self.values, dtype=pt.float32)  # shape: [T, 1]

        # Calculate the reward:
        reward_tensor = progress_tensor - pt.exp(-pt.arange(len(progress_tensor)-1,-1,-1)/10)*10*crashed
        reward_tensor = reward_tensor.to(pt.float32)
        # Calculate the advantage: reward - predicted value.
        advantage_tensor = reward_tensor - values_tensor

        if not hasattr(self, "optimizer"):
            self.optimizer = optim.Adam(self.sal.parameters(), lr=1e-8)

        device = scans_tensor.device  # Use the same device as the scans tensor.
        ppo_update(self.sal, self.optimizer,
                scans_tensor,      # states
                actions_tensor,    # actions
                log_probs_tensor,  # old_log_probs
                reward_tensor,     # returns (here, a constant reward per step)
                advantage_tensor,  # advantages
                device=device)

        self.scans.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
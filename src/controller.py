from weap_util.abstract_controller import AbstractController
from sal import SAL
import torch as pt
import torch.optim as optim
from ppo_utils import ppo_update

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

    def startup(self):

        self.values = []
        self.scans = []
        self.actions = []
        self.log_probs = []

        pass

    def compute(self, obs):
        """
        Computes control commands and returns the current set of global waypoints.
        It checks if the vehicle is near the last few waypoints and loads the next batch if needed.
        """
        scans = pt.tensor(obs["scans"][0], dtype=pt.float)[None, ...]
        dist, val = self.sal(scans)
        self.scans.append(scans)
    
        action = dist.sample()

        [[steer, speed]] = action.tolist()

        self.values.append(val.item())
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))
        # print(f"Speed: {speed:.2f}")
        print(f"Steer: {steer:.2f}")
        print(f"Mean: {dist.mean[0, 0]}\n")
        print(f"Standard Deviation: {dist.stddev[0, 0]}")

        return 1.5, steer 

    def shutdown(self):
        pass

    def train_update(self, progress, crashed):
        """
        3 Things TODO

        1. Convert the lists above into tensors using pt.cat
        2. calculate the reward which is the progress, clone that value into a tensor which is the same length as the value list
            a. Subtract ten points if it crashes
        3. calculate the advantage which is the reward - value
        4. calculate ppo update with all the tensors from above

        NOTE: Make sure we don't change the dimensions at all, the lists should be its own dimension within the tensor.
        """
        
        # Convert lists to tensors using pt.cat
        scans_tensor = pt.cat(self.scans, dim=0)          # [T, ...] observations
        actions_tensor = pt.cat(self.actions, dim=0)        # [T, ...] actions taken
        log_probs_tensor = pt.cat(self.log_probs, dim=0).sum(dim=-1)    # [T, ...] logged probabilities

        # Unsqueeze to maintain the extra dimension per timestep.
        values_tensor = pt.tensor(self.values, dtype=pt.float32).unsqueeze(1)  # shape: [T, 1]

        # Calculate the reward:
        reward_value = progress - 10 if crashed else progress
        reward_tensor = pt.full_like(values_tensor, fill_value=reward_value)  # shape: [T, 1]

        # Calculate the advantage: reward - predicted value.
        advantage_tensor = reward_tensor - values_tensor

        # @Ian do we need to do this?
        if not hasattr(self, "optimizer"):
            self.optimizer = optim.Adam(self.sal.parameters(), lr=1e-4)

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

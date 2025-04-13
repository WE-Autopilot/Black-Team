from weap_util.abstract_controller import AbstractController
from sal import SAL
import torch as pt

def addNoise(obs):
    """
    Adds noise to the observation data.

    We can add the rolling stuff here @Ian
    """
    return obs

class Controller(AbstractController):
        
    def __init__(self, path = "model.ckpt") -> None:
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
        scans = pt.tensor(obs["scans"], dtype=pt.float)
        dist, val = self.sal(scans)
        self.scans.append(scans)
    
        action = dist.sample()

        [[speed, relative_theta]] = action.tolist()
        current_theta = obs['poses_theta'][0]
        steer = relative_theta + current_theta

        self.values.append(val.item())
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))
        # print(f"Speed: {speed:.2f}")
        return 1, relative_theta

    def shutdown(self):
        pass

    def train_update(self):
        pass

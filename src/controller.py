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
        
    def __init__(self) -> None:
        super().__init__()
        self.sal = SAL()

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
        scans = pt.tensor(obs["scans"], dtype=pt.float, device=self.sal.device)
        dist, val = self.sal(scans)
        self.scans.append(scans)
    
        action = dist.sample()

        [[speed, steer]] = action.tolist()

        self.values.append(val.item())
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))

        return speed, steer

    def shutdown(self):
        pass

    def train_update(self):
        pass
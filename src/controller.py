from weap_util.abstract_controller import AbstractModel
from sal import SAL
import torch as pt

class Controller(AbstractModel):
        
    def __init__(self, path="model.ckpt") -> None:
        super().__init__()
        self.sal = SAL()
        self.sal.load(path)

    def init(self):
        pass 
    
    def startup(self):
        pass

    def eval(self, obs, timestamp=0):
        """
        Computes control commands and returns the current set of global waypoints.
        It checks if the vehicle is near the last few waypoints and loads the next batch if needed.
        """

        scans = pt.tensor(obs["scans"][0], dtype=pt.float)[None, ...]
        dist, val = self.sal(scans)
    
        action = dist.sample()

        [[steer, speed]] = action.tolist()

        return speed*7, steer 

    def shutdown(self):
        pass
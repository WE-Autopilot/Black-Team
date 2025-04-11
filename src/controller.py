from weap_util.abstract_controller import AbstractController
from sal import SAL

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
        pass

    def compute(self, obs):
        """
        Computes control commands and returns the current set of global waypoints.
        It checks if the vehicle is near the last few waypoints and loads the next batch if needed.
        """
        scans = obs["scans"][0]
        speed, steer = self.sal(scans)

        return speed, steer

    def shutdown(self):
        pass
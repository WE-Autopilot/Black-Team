from weap_util.abstract_controller import AbstractController
from weap_util.lidar import lidar_to_bitmap
from pilot import Pilot

class Controller(AbstractController):
    def setConf(self, conf_dict):
        self.conf_dict = conf_dict
        
    def startup(self):

        self.pilot = Pilot()

    def compute(self, obs):
        """
        Computes control commands and returns the current set of global waypoints.
        It checks if the vehicle is near the last few waypoints and loads the next batch if needed.
        """

        bitMap = lidar_to_bitmap(obs['scans'][0])

        speed, steer = self.pilot.get_actuation(obs, bitMap)

        return speed, steer, self.pilot.waypoints

    def shutdown():
        pass

import numpy as np
import torch as pt
from sal import SAL
from pure_pursuit import PurePursuitPlanner

def is_near_last_waypoint(position, waypoints, threshold=1.0):
    """
    Checks if the vehicle is near the 8th waypoint.
    """
    if waypoints.shape[0] < 8:
        return False
    midpoint = waypoints[7, :]  # 8th waypoint (0-indexed).
    return np.linalg.norm(position - midpoint) < threshold

def convert_to_global_waypoints(rel_waypoints, car_x, car_y, rotation, scale):
    """
    Converts relative waypoints into global coordinates.
    
    Parameters:
        - rel_waypoints: np.ndarray of shape (N,2), relative waypoints.
        - car_x, car_y: Current global coordinates of the car.
        - scale: Scaling factor (conversion from pixels to meters).
        - rotation: Rotation angle (in radians) of the car.
    
    Returns:
        - global_waypoints: np.ndarray of shape (N,2).
    """
    if rel_waypoints.ndim != 2 or rel_waypoints.shape[1] != 2:
        raise ValueError("rel_waypoints must have shape (N,2)")
    # Convert from pixels to meters.
    scaled_waypoints = rel_waypoints * scale
    # Compute the cumulative sum (tip-to-tail) to get local coordinates.
    cumsum_waypoints = np.cumsum(scaled_waypoints, axis=0)
    # Create a rotation matrix to rotate the local waypoints into the global frame.
    cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_waypoints = cumsum_waypoints @ rotation_matrix.T
    # Translate by the car's global position.
    return rotated_waypoints + np.array([car_x, car_y])

def calculate_segment_speed(self, waypoints):
    """
    Computes the constant speed for the current segment.
    For the segment from Pᵢ to Pᵢ₊₁, the speed is:
            v = ||Pᵢ₊₁ - Pᵢ|| / segment_period
    If the current segment index is invalid, returns a default speed.
    """
    idx = self._current_segment_index
    if idx is None or idx >= waypoints.shape[0] - 1:
        return 0.0
    p_current = waypoints[idx, :2]
    p_next = waypoints[idx + 1, :2]
    segment_distance = np.linalg.norm(p_next - p_current)
    return segment_distance / self.segment_period

class Pilot:
    def __init__(self):
        self.planner = PurePursuitPlanner(wheelbase=(0.17145 + 0.15875))
        self.sal = SAL(num_points=16, max_std=2)
        self.actuation = np.zeros(2)
        self.waypoints = np.empty((0, 2))

    def get_actuation(self, obs, bitMap):
        current_x = obs['poses_x'][0]
        current_y = obs['poses_y'][0]
        current_theta = obs['poses_theta'][0]

        # current_vel = pt.tensor([[np.linalg.norm([current_velx, current_vely]), current_ang_vel]], dtype=pt.float32)
        current_vel = pt.zeros(1, 2)

        position = np.array([current_x, current_y])

        if is_near_last_waypoint(position, self.waypoints):
            sal_path = self.sal(bitMap, current_vel)
            self.waypoints = convert_to_global_waypoints(sal_path, current_x, current_y, current_theta, 0.0625)
            self.actuation = self.planner.plan(current_x, current_y, current_theta, self.waypoints)
        
        return self.actuation
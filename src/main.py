import os
import yaml

import gym
import numpy as np
from PIL import Image
import pyglet

from controller import Controller
from weap_util.weap_container import run
from train_container import train_run, _render_callback
from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.f110_env import F110Env

# Monkey-patch PIL.Image.open so that it only returns the red channel (i.e. a single-channel image)
_orig_open = Image.open
Image.open = lambda *args, **kwargs: _orig_open(*args, **kwargs).convert("RGB").split()[0]

def get_track_names(maps_folder):
    """
    Returns a list of available track names based on files in the maps folder.
    Expects each track to have a _map.png, _map.yaml, and _raceline.csv.
    """
    tracks = []
    for file in os.listdir(maps_folder):
        if file.endswith(".png"):
            track_name = file.replace(".png", "")
            yaml_path = os.path.join(maps_folder, f"{track_name}.yaml")
            csv_path = os.path.join(maps_folder, f"{track_name}.csv")
            if os.path.exists(yaml_path):
                tracks.append((track_name, yaml_path, csv_path))
    return tracks

def training_mode():
    """Runs the training mode."""
    controller = Controller()
    maps_folder = "../assets/maps"
    tracks = get_track_names(maps_folder)

    first_track = tracks[0]

    env = gym.make('f110_gym:f110-v0',
                map=os.path.join(maps_folder, first_track[0]),
                map_ext=".png",
                num_agents=1,
                timestep=0.01,
                integrator=Integrator.RK4)

    while 1:
        for track_name, yaml_path, csv_path in tracks:
            print(f"\nStarting training on track: {track_name}")

            """
            ! THIS CODE SHOULD NOT WORK DO NOT TOUCH PLEAAAASSEEEE 

            BEWARE
            """
            
            obs, _, _, _ = env.reset(np.array([[0, 0, 0]]))
            env.render(mode='human')
            env.update_map(yaml_path, ".png")
            F110Env.renderer.update_obs(obs)
            env.add_render_callback(_render_callback)
            
            
            F110Env.renderer.poses = None 
            F110Env.renderer.batch = pyglet.graphics.Batch()
            F110Env.renderer.update_map(maps_folder+"/"+track_name, ".png")
            waypoints = np.loadtxt(csv_path, delimiter=",")
            starting_wpts = waypoints[::64]
            print(maps_folder+"/"+track_name)

            train_run(controller, env, maps_folder+"/"+track_name, ".png", waypoints, starting_wpts, True)
            

def normal_mode():
    """Runs the normal driving mode."""
    controller = Controller()
    run(controller, "../assets/maps/map0","map0", True)

if __name__ == "__main__":

    # Enable training mode or normal execution
    TRAIN_MODE = True  # Toggle this flag to switch between training and normal execution

    if TRAIN_MODE:
        training_mode()
    else:
        normal_mode()

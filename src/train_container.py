import time
import gym
import math
import numpy as np
import pyglet  # We assume pyglet is used by the renderer

import matplotlib.pyplot as plt
from weap_util.lidar import lidar_to_bitmap
from f110_gym.envs.base_classes import Integrator

# Global variables for the arrow rendering.
current_arrow_direction = None
rendered_arrow = []

def _render_callback(env_renderer):
    """
    Custom render callback that updates the camera and draws waypoints and heading.
    """
    e = env_renderer

    # Update camera to follow the car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 300

    e.left = left - 400
    e.right = right + 400
    e.top = top + 400
    e.bottom = bottom - 400

    # Clear the previously drawn arrow.
    global rendered_arrow
    for obj in rendered_arrow:
        obj.delete()
    rendered_arrow = []

    # Draw the arrow indicating the steering direction.
    global current_arrow_direction
    if current_arrow_direction is not None:
        # Compute the car's center.
        vertices = np.array(e.cars[0].vertices).reshape(-1, 2)
        car_center = np.mean(vertices, axis=0)

        # Define arrow length and compute the end point.
        arrow_length = 100.0
        end_x = car_center[0] + arrow_length * math.cos(current_arrow_direction)
        end_y = car_center[1] + arrow_length * math.sin(current_arrow_direction)

        arrow_obj = e.batch.add(
            2,
            pyglet.gl.GL_LINES,
            None,
            ('v3f/stream', [car_center[0], car_center[1], 0.0, end_x, end_y, 0.0]),
            ('c3B/stream', [0, 255, 0, 0, 255, 0])
        )
        rendered_arrow.append(arrow_obj)

def train_run(model, map_path, map_ext, waypoints, starting_wpts, render_on=True):
    # Create the environment.
    env = gym.make('f110_gym:f110-v0',
                   map=map_path,
                   map_ext=map_ext,
                   num_agents=1,
                   timestep=0.01,
                   integrator=Integrator.RK4)
    
    global current_arrow_direction

    for sx, sy in starting_wpts:
        model.startup()
        # Reset environment and get initial observation.
        # obs, step_reward, done, info = env.reset(np.array([[0, 0, 0]]))
        obs, step_reward, done, _ = env.reset(np.array([[sx, sy, 0]]))
        # Retrieve lap count for the ego agent.
        # The environment's lap_counts is assumed to be a list or array (one entry per agent).
        lap_count = env.lap_counts[env.ego_idx] if hasattr(env, "lap_counts") else 0
        
        # Check for termination: either a crash or when 1 lap is completed.
        if done or lap_count >= 1:
            if lap_count >= 1:
                done = True

        if render_on:
            # print("Registering render callback...")
            env.add_render_callback(_render_callback)
            env.render(mode='human')

        laptime = 0.0

        speed, steer = 0, 0

        start = time.time()

        time_limit = 10.0  # seconds

        snapshot = 0
        # Main simulation loop.
        while not done:
            snapshot += 1
            # Update the global variable for rendering.
            
            obs, step_reward, done, _ = env.step(np.array([[steer, speed]]))

            laptime += step_reward
            if render_on:
                env.render(mode='human')

            if laptime > time_limit:
                break

            speed, steer = model.compute(obs)
            # Update the arrow steering direction.
            current_theta = obs['poses_theta'][0]
            current_arrow_direction = current_theta + steer

            if snapshot > 100:
                snapshot = 0

                image = lidar_to_bitmap(scan=obs["scans"][0], channels=1, fov=2 * np.pi, draw_mode="FILL", bg_color="black", draw_center=False)

                polar_vec = lambda angle, mag: np.array([mag * np.cos(angle), mag * np.sin(angle)])

                plt.imshow(image)
                x, y = 128, 128
                dx, dy = polar_vec(current_arrow_direction, speed) * 10
                plt.scatter(x, y)
                plt.arrow(x, y, dx, dy, head_width=4, head_length=4, fc='black', ec='black')
                plt.show()


        # TRIAL FINISHED
        print("crashed" if obs["collisions"] else "done", end="\n\n\n")
        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


    def get_progress(waypoints, starting_wpts):
        """
        Returns the progress of the car from the start position.
        """
        progress = lambda wps, pos, start : np.argmin(np.linalg.norm(wps - pos, axis=1)) - start

        return progress
    

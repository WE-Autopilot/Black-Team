import time
import gym
import numpy as np

from f110_gym.envs.base_classes import Integrator

def _render_callback(env_renderer):
    """
    Custom render callback that updates the camera and renders waypoints.
    Uses the global current_waypoints_global variable for drawing.
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

def train_run(model, map_path, map_ext, waypoints, starting_wpts, render_on=True):
    # Create the environment.
    env = gym.make('f110_gym:f110-v0',
                   map=map_path,
                   map_ext=map_ext,
                   num_agents=1,
                   timestep=0.01,
                   integrator=Integrator.RK4)
    
    for sx, sy, stheta in starting_wpts:
        model.startup()
        # Reset environment and get initial observation.
        # obs, step_reward, done, info = env.reset(np.array([[0, 0, 0]]))
        obs, step_reward, done, _ = env.reset(np.array([[sx, sy, stheta]]))
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

        # Main simulation loop.
        while not done:
            # Update the global variable for rendering.
            
            obs, step_reward, done, _ = env.step(np.array([[steer, speed]]))

            laptime += step_reward
            if render_on:
                env.render(mode='human')

            if laptime > time_limit:
                break

            speed, steer = model.compute(obs)

        # TRIAL FINISHED
        print("crashed" if obs["collisions"] else "done", end="\n\n\n")
        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


    def get_progress(waypoints, starting_wpts):
        """
        Returns the progress of the car from the start position.
        """
        progress = lambda wps, pos, start : np.argmin(np.linalg.norm(wps - pos, axis=1)) - start

        return progress
    
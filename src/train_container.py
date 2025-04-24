import time
import math
import numpy as np
import pyglet

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from weap_util.lidar import lidar_to_bitmap

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

    e.left = left - 1000
    e.right = right + 1000
    e.top = top + 1000
    e.bottom = bottom - 1000

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

def train_run(model, env, map_path, map_ext, waypoints, starting_wpts, render_on=True):
    print("Loading map image from:", map_path + map_ext)
    global current_arrow_direction

    # ←–– SET UP MATPLOTLIB INTERACTIVE FIGURE ONCE
    if render_on:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,6))

        # LiDAR scan points as a scatter
        scan_scatter = ax.scatter([], [], s=5, c='cyan')

        # Steering arrow: use FancyArrowPatch so we can update it
        steer_arrow = FancyArrowPatch(
            posA=(0,0), posB=(0,0),
            arrowstyle='->', mutation_scale=20, color='red'
        )
        ax.add_patch(steer_arrow)

        # Speed label (axes‐fraction coords, top‐left)
        speed_text = ax.text(
            0.02, 0.95, '',
            transform=ax.transAxes,
            fontsize=12, verticalalignment='top'
        )

        ax.set_aspect('equal')
        max_range = 10.0  # match your LiDAR max distance
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)

    for i, (sx, sy, stheta) in enumerate(starting_wpts):
        print(f"\nStarting training on track: {map_path + map_ext} with starting position: ({sx}, {sy}, {stheta})")
        model.startup(waypoints)
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

        time_limit = 100.0  # seconds

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

            # ←–– UPDATE MATPLOTLIB PLOT EVERY 10 STEPS
            if render_on and snapshot % 10 == 0:
                # --- CORRECTED LiDAR TRANSFORM ---
                scan = obs['scans'][0]
                N = len(scan)
                fov = 2*np.pi
                # angles from -π/2 to +3π/2 so 0 is straight ahead (+Y)
                angles = np.linspace(-fov/2, fov/2, N, endpoint=False)
                # x to the right, y forward
                xs = -scan * np.sin(angles)
                ys =  scan * np.cos(angles)
                scan_scatter.set_offsets(np.column_stack((xs, ys)))

                # b) Update steering arrow in robot frame
                arrow_length = 0.5
                dx = -arrow_length * math.sin(steer)
                dy =  arrow_length * math.cos(steer)
                steer_arrow.set_positions((0,0), (dx,dy))

                # c) Update speed text
                speed_text.set_text(f"Speed: {speed:.2f}")

                # d) Redraw
                fig.canvas.draw()
                fig.canvas.flush_events()

        # TRIAL FINISHED
        print("crashed" if obs["collisions"] else "done", end="\n\n\n")
        print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)

        model.train_update(obs)

#! Deprecated
# def get_progress(waypoints, pos, start_index):
#     """
#     Computes the number of waypoints passed since the starting position.

#     Args:
#         waypoints (np.array): Array of waypoints, where each row is [x, y].
#         pos (np.array): Current position [x, y] of the car.
#         start_pos (np.array): Starting position [x, y] of the car.
    
#     Returns:
#         int: The number of waypoints passed, i.e., the difference between the closest waypoint index 
#              to the current position and the index of the waypoint closest to the starting position.
#     """
#     xy_waypoints = waypoints[:, :2]
#     current_index = np.argmin(np.linalg.norm(xy_waypoints - pos, axis=1))
#     progress_val = current_index - start_index
#     return progress_val

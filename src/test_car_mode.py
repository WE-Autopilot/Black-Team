import time
import math

import numpy as np
import torch as pt
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from sal import SAL

def run_test_car(model_path="model.ckpt", sensor_rate_hz=10):
    """
    Continuously:
      1) read a single LiDAR scan
      2) feed it through SAL to get (steer, speed)
      3) update a live Matplotlib plot of the scan + arrow + speed
    """
    # 1) Load your SAL model
    sal = SAL()
    sal.load(model_path)

    # 2) Build an interactive Matplotlib figure once
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))
    scan_scatter = ax.scatter([], [], s=5, c='cyan')
    steer_arrow = FancyArrowPatch(
        posA=(0,0), posB=(0,0),
        arrowstyle='->', mutation_scale=20, color='red'
    )
    ax.add_patch(steer_arrow)
    speed_text = ax.text(
        0.02, 0.95, '',
        transform=ax.transAxes,
        fontsize=12, va='top'
    )
    ax.set_aspect('equal')
    max_range = 10.0  # tweak to your LiDAR max
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    # 3) Main loop: grab scans, predict, plot
    dt = 1.0 / sensor_rate_hz
    while True:
        # a) Acquire one scan: shape (N,)
        scan = get_lidar_scan()

        # b) SAL expects a batch of [1, N], dtype float
        scans_tensor = pt.tensor(scan, dtype=pt.float32)[None, :]

        # c) Run through SAL → distribution
        dist, _ = sal(scans_tensor)

        # d) Use the distribution’s mean as your prediction
        steer = float(dist.mean[0,0])
        speed = float(dist.mean[0,1]) * 7  # same scaling as Controller

        # e) Convert polar→Cartesian in the *robot* frame:
        #    zero‐angle = straight ahead (+Y), right is +X
        N = len(scan)
        angles = np.linspace(-np.pi, np.pi, N, endpoint=False)
        xs = -scan * np.sin(angles)
        ys =  scan * np.cos(angles)
        scan_scatter.set_offsets(np.column_stack((xs, ys)))

        # f) Update the arrow so steer=0 → arrow points up (+Y)
        arrow_len = 0.5
        dx = -arrow_len * math.sin(steer)
        dy =  arrow_len * math.cos(steer)
        steer_arrow.set_positions((0,0), (dx,dy))

        # g) Update speed label
        speed_text.set_text(f"Speed: {speed:.2f}")

        # h) Redraw
        fig.canvas.draw()
        fig.canvas.flush_events()

        # i) Wait for next scan
        time.sleep(dt)

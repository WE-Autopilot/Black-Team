import numpy as np
import torch as pt
import PIL.Image as Image
from laser_models import ScanSimulator2D
from weap_util.lidar import lidar_to_bitmap
from glob import glob
import h5py as hp
from tqdm import tqdm
import argparse
import yaml


def gen_perp_points(curr_wp, next_wp, num_points=10, min_dis=-1.5, max_dis=1.5):
    vector = next_wp - curr_wp
    perp = np.array([-vector[1], vector[0]])
    unit_perp = perp / np.linalg.norm(perp)
    distances = np.random.uniform(min_dis, max_dis, num_points)
    points = curr_wp + unit_perp * distances[:, np.newaxis]
    return points


parser = argparse.ArgumentParser()
parser.add_argument("--num_beams", type=float, default=1080, help="Number of beams used by lidar scan.")
parser.add_argument("--fov", type=float, default=2 * np.pi, help="FOV of lidar scan.")
parser.add_argument("--interval", type=int, default=1, help="Skips by intervals when scanning waypoints.")
parser.add_argument("--dataset_path", type=str, default="dataset.h5", help="Specify path to save dataset (e.g. dataset.h5).")
parser.add_argument("--maps_path", type=str, default="maps", help="Specify path to read maps (e.g. ./).")
parser.add_argument("--num_points", type=int, default=10, help="Specify path to read maps (e.g. ./).")
parser.add_argument("--spread", type=float, default=1.45, help="Specify path to read maps (e.g. ./).")
parser.add_argument("--angle_spread", type=float, default=1.5, help="Specify path to read maps (e.g. ./).")
parser.add_argument("--lookahead", type=int, default=5, help="Specify path to read maps (e.g. ./).")
parser.add_argument("--max_noise", type=float, default=10, help="Specify max scale factor.")
args = parser.parse_args()


num_beams = args.num_beams
fov = args.fov
map_ext = ".png"
interval = args.interval
dataset_path = args.dataset_path
maps_path = args.maps_path

num_points = args.num_points
spread = args.spread
angle_spread = args.angle_spread
lookahead = args.lookahead
max_noise = args.max_noise
low_kernel = int(num_beams * 0.5)
high_kernel = int(num_beams * 0.05)
low_padding = (low_kernel - 1) - (low_kernel - 1) // 2
high_padding = (high_kernel - 1) - (high_kernel - 1) // 2

map_paths = sorted(glob(f"{maps_path}/*.yaml"))
scan_sim = ScanSimulator2D(num_beams, fov)

with hp.File(dataset_path, "w") as file:

    file.create_dataset("interval", data=interval)

    for map_path in tqdm(map_paths, desc="Scanning...", unit="map"):
        map_id = map_path.replace(f"{maps_path}/", "").replace(".yaml", "")

        with open(map_path, 'r') as conf_file:
            config = yaml.safe_load(conf_file)
        resolution = config["resolution"]

        scan_sim.set_map(map_path, map_ext)
        csv_path = map_path.replace(".yaml", ".csv")
        waypoints = np.loadtxt(csv_path, delimiter=",")[:, :2]
        waypoints = np.concatenate((waypoints, waypoints[-2:0:-1]), axis=0)

        num_waypoints = len(waypoints)
        num_startpoints = num_waypoints // interval

        file.create_group(map_id)
        map_group = file[map_id]
        map_group.create_dataset("lidar", shape=(num_startpoints * num_points, num_beams), dtype="f4", chunks=None)
        map_group.create_dataset("steer", shape=(num_startpoints * num_points,), dtype="f4", chunks=None)
        lidar_dataset = map_group["lidar"]
        steer_dataset = map_group["steer"]

        for start_ind in range(num_startpoints):

            waypoint_ind = start_ind * interval
            curr_wp = waypoints[waypoint_ind]
            next_wp = waypoints[(waypoint_ind + 1) % len(waypoints)]
            target_wp = waypoints[(waypoint_ind + lookahead) % len(waypoints)]
            wp_vec = next_wp - curr_wp
            track_angle = np.arctan2(*wp_vec[::-1])

            points = gen_perp_points(curr_wp, next_wp, num_points=num_points, min_dis=-spread, max_dis=spread)
            dis_angles = np.random.uniform(-angle_spread, angle_spread, num_points)
            low_noises = pt.nn.functional.avg_pool1d(max_noise / 2 * pt.rand(num_points, num_beams), low_kernel, 1, low_padding).numpy()[:, :num_beams]
            high_noises = pt.nn.functional.avg_pool1d(max_noise * (2 * pt.rand(num_points, num_beams) - 1), high_kernel, 1, high_padding).numpy()[:, :num_beams]
            noises = low_noises + high_noises
            noises = noises * (noises >= 0)

            for point_ind, point, dis_angle, noise in zip(range(num_points), points, dis_angles, noises):
                pos_angle = track_angle + dis_angle
                pose = np.append(point, pos_angle)
                path_vec = target_wp - point
                path_angle = np.arctan2(*path_vec[::-1])

                steer = path_angle - pos_angle
                sign = steer / abs(steer)
                steer = abs(steer) % (2 * np.pi)
                steer = sign * steer if steer < np.pi else -sign * (2 * np.pi - steer)

                scan = scan_sim.scan(pose, np.random.default_rng()) + noise

                lidar_dataset[start_ind * num_points + point_ind] = scan
                steer_dataset[start_ind * num_points + point_ind] = steer

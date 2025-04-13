import numpy as np
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
parser.add_argument("--maps_path", type=str, default="./", help="Specify path to read maps (e.g. ./).")
args = parser.parse_args()


num_beams = args.num_beams
fov = args.fov
map_ext = ".png"
interval = args.interval
dataset_path = args.dataset_path
maps_path = args.maps_path

num_points = 10
spread = 1.5
lookahead = 5

map_paths = sorted(glob(f"{maps_path}maps/*.yaml"))
scan_sim = ScanSimulator2D(num_beams, fov)

with hp.File(dataset_path, "w") as file:

    file.create_dataset("interval", data=interval)

    for map_path in tqdm(map_paths, desc="Scanning...", unit="map"):
        map_id = map_path.replace(f"{maps_path}maps/", "").replace(".yaml", "")

        with open(map_path, 'r') as conf_file:
            config = yaml.safe_load(conf_file)
        resolution = config["resolution"]

        scan_sim.set_map(map_path, map_ext)
        csv_path = map_path.replace(".yaml", ".csv")
        waypoints = np.loadtxt(csv_path, delimiter=",")

        num_waypoints = len(waypoints)
        num_startpoints = num_waypoints // interval

        file.create_dataset(map_id, shape=(num_startpoints * num_points, num_beams), dtype="f4", chunks=None)
        scan_dataset = file[map_id]

        for start_ind in range(num_startpoints):

            waypoint_ind = start_ind * interval
            curr_wp = waypoints[waypoint_ind]
            next_wp = waypoints[(waypoint_ind + lookahead) % len(waypoints)]

            points = gen_perp_points(curr_wp, next_wp, num_points=10, min_dis=-1.5, max_dis=1.5)

            for point_ind, point in enumerate(points):
                path_vec = next_wp - point
                angle = np.arctan2(*path_vec[::-1])
                pose = np.append(point, angle)

                scan = scan_sim.scan(pose, np.random.default_rng())

                scan_dataset[start_ind * num_points + point_ind] = scan

import numpy as np
import PIL.Image as Image
from laser_models import ScanSimulator2D
from weap_util.lidar import lidar_to_bitmap
from glob import glob
import h5py as hp
from tqdm import tqdm
import argparse
import yaml

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

            points = np.random.uniform(-1, 1, (num_points, 2))
            points = points / np.linalg.norm(points, axis=1).max() * spread / resolution + waypoints[waypoint_ind]

            for point_ind, point in enumerate(points):
                path_vec = waypoints[(waypoint_ind + 1) % len(waypoints)] - point
                angle = np.arctan2(*path_vec)
                pose = np.append(waypoints[waypoint_ind], angle)

                scan = scan_sim.scan(pose, np.random.default_rng())

                scan_dataset[start_ind * num_points + point_ind] = scan

import numpy as np
import PIL.Image as Image
from laser_models import ScanSimulator2D
from weap_util.lidar import lidar_to_bitmap
from glob import glob
import h5py as hp
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_beams", type=float, default=1080, help="Number of beams used by lidar scan.")
parser.add_argument("--fov", type=float, default=2 * np.pi, help="FOV of lidar scan.")
parser.add_argument("--interval", type=int, default=1, help="Skips by intervals when scanning waypoints.")
parser.add_argument("--dataset_path", type=string, default="dataset.h5", help="Specify path to save dataset (e.g. dataset.h5).")
parser.add_argument("--maps_path", type=string, default="./", help="Specify path to read maps (e.g. ./).")
args = parser.parse_args()


num_beams = args.num_beams
fov = args.fov
map_ext = args.map_ext
interval = args.interval
dataset_path = args.dataset_path
maps_path = args.maps_path

map_paths = sorted(glob(f"{maps_path}maps/*.yaml"))
scan_sim = ScanSimulator2D(num_beams, fov)

with hp.File(dataset_path, "w") as file:

    file.create_dataset("interval", data=interval)

    for map_path in tqdm(map_paths, desc="Scanning...", unit="map"):
        map_id = map_path.replace(f"{maps_path}maps/", "").replace(".yaml", "")

        file.create_group(map_id)
        file_group = file[map_id]

        scan_sim.set_map(map_path, map_ext)
        csv_path = f"{maps_path}centerline/{map_id}.csv"
        waypoints = np.loadtxt(csv_path, delimiter=",")
        path_vecs = np.roll(waypoints, -1, axis=0) - waypoints

        num_waypoints = len(waypoints)
        num_startpoints = num_waypoints // interval

        file_group.create_dataset("paths", data=path_vecs, chunks=None)
        file_group.create_dataset("lidar", shape=(num_startpoints, 256, 256), dtype="i8", chunks=None)
        lidar_dataset = file_group["lidar"]

        for start_ind in range(num_startpoints):
            waypoint_ind = start_ind * interval

            angle = np.arctan2(*path_vecs[waypoint_ind, ::-1])
            pose = np.append(waypoints[waypoint_ind], angle)

            scan = scan_sim.scan(pose, np.random.default_rng())
            lidar_img = lidar_to_bitmap(scan=scan, channels=1, fov=fov, draw_mode='FILL', bg_color='black', draw_center=False)
            lidar_bitmap = lidar_img / 255

            lidar_dataset[start_ind] = lidar_bitmap

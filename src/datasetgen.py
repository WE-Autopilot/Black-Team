import numpy as np
import PIL.Image as Image
from laser_models import ScanSimulator2D
from weap_util.lidar import lidar_to_bitmap
from glob import glob
import h5py as hp
from tqdm import tqdm


num_beams = 1080
fov = 2 * np.pi
map_ext = '.png'
interval = 10
dataset_path = "dataset.h5"

map_paths = sorted(glob("maps/*.yaml"))
scan_sim = ScanSimulator2D(num_beams, fov)

with hp.File(dataset_path, "w") as file:

    file.create_dataset("interval", data=interval)

    for map_path in tqdm(map_paths, desc="Scanning...", unit="map"):
        map_id = map_path.replace("maps/", "").replace(".yaml", "")

        file.create_group(map_id)
        file_group = file[map_id]

        scan_sim.set_map(map_path, map_ext)
        csv_path = f"centerline/{map_id}.csv"
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

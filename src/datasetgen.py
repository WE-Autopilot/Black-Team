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
interval = 1
dataset_path = "dataset.h5"

map_paths = sorted(glob("maps/*.yaml"))
scan_sim = ScanSimulator2D(num_beams, fov)

with hp.File(dataset_path, "w") as file:

    for map_path in map_paths[:1]:

        scan_sim.set_map(map_path, map_ext)
        csv_path = map_path.replace("maps/", "centerline/").replace(".yaml", ".csv")
        waypoints = np.loadtxt(csv_path, delimiter=",")
        path_vecs = np.roll(waypoints, -1, axis=0) - waypoints

        for wp_ind in tqdm(range(0, len(waypoints), interval), desc="Travelling...", unit="waypoint"):
            theta = np.arctan2(*path_vecs[wp_ind, ::-1])
            pose = np.append(waypoints[wp_ind], theta)

            scan = scan_sim.scan(pose, np.random.default_rng())
            lidar_img = lidar_to_bitmap(scan=scan, channels=1, fov=fov, draw_mode='FILL', bg_color='black', draw_center=False)
            lidar_bitmap = lidar_img / 255
            Image.fromarray(lidar_img[::-1]).save(f"test/{wp_ind}.png")

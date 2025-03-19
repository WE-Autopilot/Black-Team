import numpy as np
import PIL.Image as Image
from laser_models import ScanSimulator2D
from weap_util.lidar import lidar_to_bitmap
from glob import glob
import h5py as hp


num_beams = 1080
fov = 4.7
map_ext = '.png'
interval = 1
dataset_path = "daataset.h5"

map_paths = sorted(glob("maps/*.yaml"))
scan_sim = ScanSimulator2D(num_beams, fov)

with hp.File(dataset_path, "w") as file:

    for map_path in map_paths:

        scan_sim.set_map(map_path, map_ext)
        csv_path = map_path.replace("maps/", "centerline/").replace(".yaml", ".csv")
        waypoints = np.loadtxt(csv_path, delimiter=",")
        path_vecs = np.roll(waypoints, -1) - waypoints

        for wp_ind in range(0, len(waypoints), interval):
            theta = np.arctan2(*path_vecs[wp_ind, ::-1])
            pose = np.append(waypoints[wp_ind], theta)

            scan = scan_sim.scan(pose, np.random.default_rng())
            lidar_img = lidar_to_bitmap(scan=scan, channels=1, fov=fov, draw_mode='FILL', bg_color='black', draw_center=False)
            lidar_bitmap = lidar_img / 255


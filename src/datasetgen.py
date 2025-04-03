import numpy as np
import PIL.Image as Image
from laser_models import ScanSimulator2D
from weap_util.lidar import lidar_to_bitmap
from glob import glob
import h5py as hp
from tqdm import tqdm
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_beams", type=float, default=1080, help="Number of beams used by lidar scan.")
parser.add_argument("--fov", type=float, default=2 * np.pi, help="FOV of lidar scan.")
parser.add_argument("--interval", type=int, default=1, help="Skips by intervals when scanning waypoints.")
parser.add_argument("--dataset_path", type=str, default="dataset.h5", help="Specify path to save dataset (e.g. dataset.h5).")
parser.add_argument("--maps_path", type=str, default="./", help="Specify path to read maps (e.g. ./).")
args = parser.parse_args()


# Initialize configuration parameters from command line arguments
num_beams = args.num_beams  # Number of beams for the lidar scanner
fov = args.fov  # Field of view for the lidar scanner (default: 360 degrees)
map_ext = ".png"  # Extension for map files
interval = args.interval  # Interval for sampling waypoints
dataset_path = args.dataset_path  # Output path for the dataset file
maps_path = args.maps_path  # Input path for map files

# Find all map YAML files in the specified directory
map_paths = sorted(glob(f"{maps_path}maps/*.yaml"))
# Initialize the 2D scan simulator with specified parameters
scan_sim = ScanSimulator2D(num_beams, fov)

# Create HDF5 file to store the dataset
with hp.File(dataset_path, "w") as file:

    # Store the sampling interval in the dataset
    file.create_dataset("interval", data=interval)

    # Process each map file
    for map_path in tqdm(map_paths, desc="Scanning...", unit="map"):
        # Extract map ID from the path
        map_id = map_path.replace(f"{maps_path}maps/", "").replace(".yaml", "")

        # Create a group in the HDF5 file for this map
        file.create_group(map_id)
        file_group = file[map_id]

        # Configure the simulator with the current map
        scan_sim.set_map(map_path, map_ext)
        # Load centerline waypoints for the current map
        csv_path = f"{maps_path}centerline/{map_id}.csv"
        waypoints = np.loadtxt(csv_path, delimiter=",")
        # Calculate direction vectors between consecutive waypoints
        path_vecs = np.roll(waypoints, -1, axis=0) - waypoints

        # Calculate number of waypoints and starting points based on interval
        num_waypoints = len(waypoints)
        num_startpoints = num_waypoints // interval

        # Store path vectors in the dataset
        file_group.create_dataset("paths", data=path_vecs, chunks=None)
        # Create dataset for lidar images (256x256 pixels)
        file_group.create_dataset("lidar", shape=(num_startpoints, 256, 256), dtype="i8", chunks=None)
        lidar_dataset = file_group["lidar"]

        # Generate lidar scans for each starting point
        for start_ind in range(num_startpoints):
            waypoint_ind = start_ind * interval

            # Calculate orientation angle based on path vector
            angle = np.arctan2(*path_vecs[waypoint_ind, ::-1])
            # Combine position and orientation into a pose
            pose = np.append(waypoints[waypoint_ind], angle)

            # Generate lidar scan from the current pose
            scan = scan_sim.scan(pose, np.random.default_rng())
            # Convert scan to bitmap image representation
            # Note: max_scan_radius is approximately 12 to 15
            lidar_img = lidar_to_bitmap(scan=scan, channels=1, fov=fov, draw_mode='FILL', bg_color='black', draw_center=False)
            # Normalize pixel values to range [0, 1]
            lidar_bitmap = lidar_img / 255

            # Store the lidar bitmap in the dataset
            lidar_dataset[start_ind] = lidar_bitmap

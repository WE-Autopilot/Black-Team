import h5py as hp
import numpy as np
import torch as pt
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from weap_util.lidar import lidar_to_bitmap


def rotate_path(path, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    rotation_matrix = np.array([[cos_theta, sin_theta],
                                    [-sin_theta, cos_theta]])

    rotated_path = np.dot(path, rotation_matrix)
    return rotated_path


def show_path(img, paths):
    plt.imshow(img[0], origin="lower")

    for path in paths:
        waypoints = np.cumsum(np.append(np.zeros(2), path).reshape(-1, 2), axis=-2) * 10 + np.array([128, 128])
        plt.plot(*waypoints.T)

    plt.show()


class Stage0Dataset(Dataset):
    def __init__(self, h5_path, path_inds=np.arange(16)):
        """Dataset class for loading and rotating LiDAR scans.
    
    This dataset loads LiDAR scans from an HDF5 file and creates augmented data
    by rotating each scan through a range of angles. For each scan in the dataset,
    it generates multiple rotated versions based on the min_rotation and max_rotation parameters.
    """
    
    def __init__(self, h5_path: str, min_index_shift: int = -256, max_index_shift: int = 256, dataset_name: str = "train"):
        """Initialize the dataset.
        
        Args:
            h5_path: Path to the HDF5 file containing LiDAR scans
            min_index_shift: Minimum index shift to apply (default: 0)
            max_index_shift: Maximum index shift to apply (default: 1080)
            dataset_name: Name of the dataset in the HDF5 file (default: 'train')
        """
        super().__init__()
        self.h5_path = h5_path
        self.min_index_shift = min_index_shift
        self.max_index_shift = max_index_shift
        self.dataset_name = dataset_name
        self.file = hp.File(self.h5_path, "r")
        
        # Get all available map datasets
        self.map_datasets = list(self.file.keys())[1:]
        
        # Calculate total number of scans across all maps
        self.map_sizes = []
        self.total_scans = 0
        for map_name in self.map_datasets:
            map_size = len(self.file[map_name])
            self.map_sizes.append(map_size)
            self.total_scans += map_size
        
        # Store cumulative sums for efficient indexing
        self.cumulative_sizes = np.cumsum([0] + self.map_sizes)
            
    def __len__(self):
        """Return the total size of the dataset.
        
        Returns:
            Total number of items (total_scans * possible_rotations)
        """
        return self.total_scans * (self.max_index_shift - self.min_index_shift)

    def __getitem__(self, index: int):
        """Get a rotated LiDAR scan.
        
        Args:
            index: Index into the dataset
            
        Returns:
            Tuple containing:
                - rotated_scan: Torch tensor of the rotated LiDAR scan
                - angle: Angle corresponding to the index shift (in radians)
        """
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of range [0, {len(self)})")
            
        # Calculate scan index and index shift
        scan_idx = index // (self.max_index_shift - self.min_index_shift)
        index_shift = self.min_index_shift + (index % (self.max_index_shift - self.min_index_shift))
        
        # Find which map dataset contains this scan
        map_idx = np.searchsorted(self.cumulative_sizes, scan_idx, side='right') - 1
        local_scan_idx = scan_idx - self.cumulative_sizes[map_idx]
        map_name = self.map_datasets[map_idx]
        
        # Get the LiDAR scan using the context manager
        lidar = self.file[map_name][local_scan_idx][:]
        
        # Convert to PyTorch tensor
        lidar_tensor = pt.tensor(lidar, dtype=pt.float)
            
        # Apply rotation using index shift
        rotated_scan = pt.roll(lidar_tensor, shifts=index_shift, dims=0)
        
        # Convert index shift to angle in radians
        angle = pt.tensor(2 * np.pi * index_shift / len(lidar_tensor))
        
        return rotated_scan, angle

    def __del__(self):
        """Cleanup method as a backup to ensure HDF5 file is properly closed.
        
        Note: This is a backup mechanism. Users should explicitly call close()
        when done with the dataset.
        """
        self.file.close()


if __name__ == "__main__":

    dataset = Stage0Dataset("dataset.h5")
    scan, angle = dataset[0]  

    print(f"Dataset length: len(dataset)")
    print(scan.shape)  
    print(angle.shape)

    for i in range(len(dataset)):
        scan, angle = dataset[i]
        print(f"{angle / pt.pi * 180:.2f} deg")
        lidar_img = lidar_to_bitmap(scan=scan, channels=1, fov=2 * pt.pi, draw_mode='FILL', bg_color='black', draw_center=False)
        plt.imshow(lidar_img, origin="lower")
        plt.show()



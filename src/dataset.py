import h5py as hp
import numpy as np
import torch as pt
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from weap_util.lidar import lidar_to_bitmap


polar_vec = lambda angle, mag: np.array([mag * np.cos(angle), mag * np.sin(angle)])


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
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self.file = hp.File(h5_path, 'r')
        self.dataset_names = list(self.file.keys())
        self.dataset_names.remove("interval")
        self.dataset_lengths = [len(self.file[name]["steer"]) for name in self.dataset_names]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, ind):
        dataset_ind = np.searchsorted(self.cumulative_lengths, ind, side='right')

        if dataset_ind == 0:
            dataset_inner_ind = ind
        else:
            dataset_inner_ind = ind - self.cumulative_lengths[dataset_ind - 1]

        dataset_name = self.dataset_names[dataset_ind]
        lidar = self.file[dataset_name]["lidar"][dataset_inner_ind]
        steer = self.file[dataset_name]["steer"][dataset_inner_ind]

        return pt.tensor(lidar, dtype=pt.float), pt.tensor(steer, dtype=pt.float)

    def __del__(self):
        self.file.close()


if __name__ == "__main__":

    dataset = Stage0Dataset("dataset.h5")
    scan, angle = dataset[0]  

    print(f"Dataset length: {len(dataset)}")
    print(scan.shape)  
    print(angle.shape)

    for i in range(len(dataset) // 127):
        scan, angle = dataset[i * 127]
        print(f"{angle / pt.pi * 180:.2f} deg")
        lidar_img = lidar_to_bitmap(scan=scan, channels=1, fov=2 * pt.pi, draw_mode='FILL', bg_color='black', draw_center=False)
        plt.imshow(lidar_img, origin="lower")
        plt.scatter(128, 128)
        x, y = 128, 128
        dx, dy = polar_vec(np.pi / 2 + angle, 10)
        plt.arrow(x, y, dx, dy, head_width=2, head_length=2, fc='black', ec='black')
        plt.show()



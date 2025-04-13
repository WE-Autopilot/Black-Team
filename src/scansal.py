import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import h5py as hp
import numpy as np
import torch as pt
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class ScanSAL(nn.Module):
    def __init__(self, num_points=16, min_std=1e-4, max_std=2):
        super().__init__()
        self.num_points = num_points
        self.min_std = min_std
        self.max_std = max_std

        # Fully connected layers using Sequential
        self.fc_layers = nn.Sequential(
            nn.Linear(1080, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * num_points),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        mean = x[:, :2 * self.num_points]
        std = F.sigmoid(x[:, 2 * self.num_points:]) * self.max_std + self.min_std
        dist = Normal(mean, std)
        return dist

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


class ScanDataset(Dataset):
    def __init__(self, h5_path, path_inds=np.arange(16)):
        super().__init__()
        self.path_inds = path_inds
        self.h5_path = h5_path
        self.file = hp.File(h5_path, 'r')
        self.interval = self.file["interval"][()]
        self.dataset_names = list(self.file.keys())
        self.dataset_names.remove("interval")
        self.dataset_lengths = [len(self.file[name]["lidar"]) for name in self.dataset_names]
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
        dataset_length = self.dataset_lengths[dataset_ind]
        num_path = len(self.file[dataset_name]["paths"])
        path_start = dataset_inner_ind * self.interval
        path_inds = (self.path_inds + path_start) % num_path
        #print(path_inds)

        #print(dataset_name, dataset_inner_ind)
        lidar = self.file[dataset_name]["scan"][dataset_inner_ind]
        raw_path = self.file[dataset_name]["paths"][:][path_inds]

        angle = np.arctan2(*raw_path[0])
        path = rotate_path(raw_path, angle)
        path_vec = path.reshape(-1)

        return pt.tensor(lidar, dtype=pt.float), pt.tensor(path_vec, dtype=pt.float)

    def __del__(self):
        self.file.close()

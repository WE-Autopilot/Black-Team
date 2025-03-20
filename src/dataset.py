import h5py as hp
import numpy as np
import torch as pt
from torch.utils.data import Dataset


def rotate_path(path, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    rotation_matrix = np.array([[cos_theta, sin_theta],
                                    [-sin_theta, cos_theta]])

    rotated_path = np.dot(path, rotation_matrix)
    return rotated_path


class H5Dataset(Dataset):
    def __init__(self, h5_path, path_inds=np.arange(16)):
        super().__init__()
        self.path_inds = path_inds
        self.h5_path = h5_path
        self.file = hp.File(h5_path, 'r')
        self.dataset_names = list(self.file.keys())
        self.dataset_lengths = [len(self.file[name]["paths"]) for name in self.dataset_names]
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
        path_inds = (self.path_inds + dataset_inner_ind) % dataset_length

        raw_lidar = self.file[dataset_name]["lidar"][dataset_inner_ind]
        raw_path = self.file[dataset_name]["paths"][path_inds]

        lidar = raw_lidar[None, ...]

        angle = np.arctan2(*raw_path[0])
        path = rotate_path(raw_path, angle)
        path_vec = path.reshape(-1)

        return pt.tensor(lidar, dtype=pt.float), pt.tensor(path_vec, dtype=pt.float)

    def __del__(self):
        self.file.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = H5Dataset("dataset.h5")
    img, path = dataset[118]  

    print(len(dataset))
    print(img.shape)  
    print(path.shape)

    waypoints = np.cumsum(np.append(np.zeros(2), path).reshape(-1, 2), axis=0) * 10 + np.array([128, 128])
    plt.imshow(img[0], origin="lower")
    plt.plot(*waypoints.T)
    plt.show()

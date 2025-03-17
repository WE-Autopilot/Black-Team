import h5py
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_file):
        super().__init__()
        self.h5_file = h5_file
        self.data = h5py.File(h5_file, 'r')
        self.images = self.data['img']  
        self.paths = self.data['path']  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")

        image = torch.tensor(self.images[idx]).unsqueeze(0)  
        path = torch.tensor(self.paths[idx])  
        
        return image, path

if __name__ == "__main__":
    dataset = H5Dataset("dummy.h5")
    img, path = dataset[0]  
    print(img.shape)  
    print(path.shape)
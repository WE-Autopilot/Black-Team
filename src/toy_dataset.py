import torch
from torch.utils.data import Dataset 

class PixelIndexer(Dataset):
    def __init__(self, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.total_pixels = width * height

    def __getitem__(self, idx):  
        if idx < 0 or idx >= self.total_pixels:
            raise IndexError("Index out of bounds")

        # Compute x, y coordinates
        y = idx // self.width
        x = idx % self.width

        # Create one-hot encoded matrix
        one_hot = torch.zeros(self.height, self.width)
        one_hot[y, x] = 1

        # Return one-hot matrix and (x, y) coordinate tensor
        return one_hot, torch.tensor([x, y])

if __name__ == "__main__":
    indexer = PixelIndexer(4, 3)
    matrix, coords = indexer[5] 
    print(matrix)
    print(coords)

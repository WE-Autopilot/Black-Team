import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, len=16):
        super().__init__()
        self.len = int(len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return F.one_hot(pt.tensor(idx), self.len).float(), pt.tensor(idx, dtype=pt.float)


class Model(nn.Module):
    def __init__(self, len):
        super().__init__()
        self.len = len
        self.fc = nn.Linear(len, 3)
                                

    def forward(self, x):
        y = self.fc(x)
        mean, std, value = y.T
        std = pt.exp(std)
        mean = F.sigmoid(mean) * self.len
        dist = Normal(mean, std)
        return dist, value




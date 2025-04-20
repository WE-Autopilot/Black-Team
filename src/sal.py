import torch.nn as nn
import torch.nn.functional as F
import torch as pt
from torch.distributions import Normal


class SAL(nn.Module):
    def __init__(self, num_beams=1080, min_std=1e-2, max_std=2, device="cpu"):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.device = device

        # Fully connected layers using Sequential
        self.fc_layers = nn.Sequential(
            nn.Linear(num_beams, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
        )

        self.to(device)

    def forward(self, x):
        x = self.fc_layers(x)
        mean = x[:, :2]
        std = F.sigmoid(x[:, 2:-1]) * self.max_std
        std = pt.clamp(std, min=self.min_std)
        dist = Normal(mean, std)
        return dist, x[:, -1]

    def load(self, path):
        state_dict = pt.load(path, map_location=self.device)
        self.load_state_dict(state_dict)

    def save(self, path):
        pt.save(self.state_dict(), path)



if __name__ == "__main__":
    sal = SAL(1080)
    img = pt.randn(16, 1080)
    d, v = sal(img)
    print(d.sample().shape, v.shape)

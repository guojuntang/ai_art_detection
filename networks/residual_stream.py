import torch.nn as nn
from .noise_3down import *

class ResidualStream(nn.Module):
    def __init__(self):
        super(ResidualStream, self).__init__()
        self.noise = Trans_Noise()
        self.n_elayers = nn.Sequential(
            residual(256, 256),
            residual(256, 256)
        )
        self.fc1 = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.noise(x)
        x = self.n_elayers(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size()[0], -1)
        x = self.fc1(x)
        return x
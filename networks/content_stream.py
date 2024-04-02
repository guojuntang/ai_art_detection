import torch.nn as nn
from .rgb_3down import *

class ContentStream(nn.Module):
    def __init__(self):
        super(ContentStream, self).__init__()
        self.rgb = Pre2()
        self.r_elayers = nn.Sequential(
            plain(256, 256),
            plain(256, 256)
        )
        self.fc1 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.rgb(x)
        x = self.r_elayers(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size()[0], -1)
        x = self.fc1(x)
        return x
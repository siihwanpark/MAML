import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )
    
    def forward(self, x):
        return self.layers(x)
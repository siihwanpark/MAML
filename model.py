import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1,40),
            nn.ReLU(),
            nn.Linear(40,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )
    
    def forward(self, x):
        return self.layers(x)

    def forward_with_params(self, x, params = None):
        # x : a tensor with size 1
        # params : an ordered dictionary of parameters
        if params == None:
            params = self.state_dict()
        
        x = F.relu(F.linear(x, params['layers.0.weight'], params['layers.0.bias']))
        x = F.relu(F.linear(x, params['layers.2.weight'], params['layers.2.bias']))
        x = F.linear(x, params['layers.4.weight'], params['layers.4.bias'])

        return x
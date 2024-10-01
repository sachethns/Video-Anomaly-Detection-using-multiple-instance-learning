import torch
import torch.nn as nn
from torch.nn import functional as F

# Loaded AUC: 0.836053143037044
# loss ~50

class Learner3(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner3, self).__init__()
        
        # Path 1 with ReLU and Tanh activations
        self.path1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.Tanh(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1)
        )
        
        # Path 2 with LeakyReLU and Sigmoid activations
        self.path2 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.Sigmoid(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1)
        )
        
        self.drop_p = drop_p
        self.weight_init()
        self.vars = nn.ParameterList()
        
        # Collect parameters for custom handling
        for path in [self.path1, self.path2]:
            for i, param in enumerate(path.parameters()):
                self.vars.append(param)

    def weight_init(self):
        for path in [self.path1, self.path2]:
            for layer in path:
                if type(layer) == nn.Linear:
                    nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        # Process input through both paths
        x1 = self.path1(x)
        x2 = self.path2(x)
        
        # Average the outputs from both paths
        return torch.sigmoid((x1 + x2) / 2.)

    def parameters(self):
        return self.vars
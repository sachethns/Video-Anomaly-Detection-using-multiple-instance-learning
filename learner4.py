import torch
import torch.nn as nn
from torch.nn import functional as F

# Loaded AUC: 0.8449675643003246
# loss ~ 0.25

class Learner4(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner4, self).__init__()
        
        # Main Path
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, input_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.drop_p = drop_p
        self.weight_init()
        self.vars = nn.ParameterList()
        
        # Collect parameters
        for module in [self.main_path, self.classifier]:
            for i, param in enumerate(module.parameters()):
                self.vars.append(param)

    def weight_init(self):
        for module in [self.main_path, self.classifier]:
            for layer in module:
                if type(layer) == nn.Linear:
                    nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        residual = x
        x = self.main_path(x)
        x += residual
        return self.classifier(x)

    def parameters(self):
        return self.vars
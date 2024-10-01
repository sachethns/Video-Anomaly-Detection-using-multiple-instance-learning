import torch
import torch.nn as nn
from torch.nn import functional as F

# Loaded AUC: 0.8284667201938799
# loss-0.73

class Learner6(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner6, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(drop_p),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(drop_p),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(drop_p),

            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.drop_p = drop_p
        self.weight_init()
        self.vars = nn.ParameterList()
        self.filter1 = nn.LayerNorm(input_dim)

        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        
        return self.classifier(x)
    

    def parameters(self):
        return self.vars

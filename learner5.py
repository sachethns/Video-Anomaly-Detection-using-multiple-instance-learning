import torch
import torch.nn as nn
from torch.nn import functional as F



class Learner5(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner5, self).__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.drop_p = drop_p
        self.weight_init()
        self.vars = nn.ParameterList()
        
        # Collect parameters
        for module in [self.attention, self.classifier]:
            for i, param in enumerate(module.parameters()):
                self.vars.append(param)

    def weight_init(self):
        for module in [self.attention, self.classifier]:
            for layer in module:
                if type(layer) == nn.Linear:
                    nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars

        attn_weights = self.attention(x)
        x = attn_weights * x
        return self.classifier(x)

    def parameters(self):
        return self.vars

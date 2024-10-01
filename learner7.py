import torch
import torch.nn as nn

# auc = {} 0.8411476410833673
# loss 0.25

class ResidualBlock(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, input_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.linear3(out)
        
        # Ensure out and identity have the same shape before addition
        if out.shape != identity.shape:
            identity = self.downsample(identity)
        
        out += identity
        return self.relu(out)

class LearnerResidual(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(LearnerResidual, self).__init__()
        
        # Define a couple of residual blocks
        self.residual1 = ResidualBlock(input_dim=input_dim, drop_p=drop_p)
        self.residual2 = ResidualBlock(input_dim=input_dim, drop_p=drop_p)
        
        # Final classifier to produce output
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

    def forward(self, x):
        x = self.residual1(x)
        x = self.residual2(x)
        return self.classifier(x)

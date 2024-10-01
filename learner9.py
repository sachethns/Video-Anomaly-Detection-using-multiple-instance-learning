import torch
import torch.nn as nn
from torch.nn import functional as F

# auc 0.82
# loss 0.84


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            SelfAttention(input_dim) for _ in range(num_heads)
        ])

    def forward(self, x):
        outputs = [head(x) for head in self.attention_heads]
        return torch.mean(torch.stack(outputs), dim=0)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_weights = self.softmax(torch.matmul(Q, K.t()))
        output = torch.matmul(attention_weights, V)
        
        return output

class Learner9(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner9, self).__init__()

        self.attention = MultiHeadSelfAttention(input_dim, num_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(1024, 512),
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
        self.filter1 = nn.LayerNorm(input_dim)

    def weight_init(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        # Apply attention mechanism
        att_x = self.attention(x)

        x1 = self.classifier[:5](x + att_x) # Adding skip connection
        x2 = F.relu(self.filter1(x))
        x2 = self.classifier[:5](x2 + att_x) # Adding skip connection

        x = (x1 + x2) / 2.
        x = self.classifier[5:](x) 

        return torch.sigmoid(x)
    




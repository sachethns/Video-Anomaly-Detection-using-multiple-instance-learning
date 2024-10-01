import torch
import torch.nn as nn
import torch.nn.functional as F

# Loaded AUC: 0.8459241558417792
# loss 0.21

class StreamAttention(nn.Module):
    def __init__(self, input_dim):
        super(StreamAttention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 3)

    def forward(self, x):
        return F.softmax(self.attention_weights(x), dim=1)

class Learner8(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner8, self).__init__()

        # Stream 1
        self.stream1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1)
        )
        
        # Stream 2
        self.stream2 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1)
        )
        
        # Stream 3
        self.stream3 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(drop_p),
            nn.Linear(512, 32),
            nn.LeakyReLU(),
            nn.Dropout(drop_p),
            nn.Linear(32, 1)
        )
        
        # Stream attention mechanism
        self.stream_attention = StreamAttention(input_dim)

        self.weight_init()
        self.vars = nn.ParameterList()
        
        # Collect parameters from all streams
        for stream in [self.stream1, self.stream2, self.stream3, self.stream_attention]:
            for param in stream.parameters():
                self.vars.append(param)

    def weight_init(self):
        for stream in [self.stream1, self.stream2, self.stream3]:
            for layer in stream:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        if vars is None:
            vars = self.vars
        
        x1 = self.stream1(x)
        x2 = self.stream2(x)
        x3 = self.stream3(x)

        # Get attention weights for each stream
        attention_weights = self.stream_attention(x)
        
        # Combine streams based on attention weights
        combined_output = attention_weights[:, 0:1] * x1 + attention_weights[:, 1:2] * x2 + attention_weights[:, 2:3] * x3
        return torch.sigmoid(combined_output)

    def parameters(self):
        return self.vars

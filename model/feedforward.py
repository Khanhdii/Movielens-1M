import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

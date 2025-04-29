import torch.nn as nn
from .attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, num_heads):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding.embedding_dim, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.self_attn = MultiHeadAttention(hidden_size * 2, num_heads)
        self.layernorm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        outputs = outputs + self.self_attn(outputs, outputs, outputs)[0]
        outputs = self.layernorm(outputs)
        return outputs, (hidden, cell)

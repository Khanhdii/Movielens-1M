import torch.nn as nn
from .attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, num_heads):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(embedding.embedding_dim, hidden_size, num_layers=2, batch_first=True)
        self.cross_attn = MultiHeadAttention(hidden_size, num_heads)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.encoder_proj = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, tgt, encoder_outputs, hidden, cell):
        embedded = self.dropout(self.embedding(tgt))
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        encoder_outputs = self.encoder_proj(encoder_outputs)
        outputs = outputs + self.cross_attn(outputs, encoder_outputs, encoder_outputs)[0]
        outputs = self.layernorm(outputs)
        prediction = self.fc_out(outputs)
        return prediction, hidden, cell

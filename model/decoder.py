import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.positional_encoding import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feedforward = FeedForward(d_model, dim_feedforward, dropout)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.pos_encoder(tgt)
        
        # Self-attention
        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.layer_norm1(tgt + self.dropout(self_attn_output))
        
        # Cross-attention
        cross_attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.layer_norm2(tgt + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feedforward(tgt)
        tgt = self.layer_norm3(tgt + self.dropout(ff_output))
        
        return tgt

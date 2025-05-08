import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        q = self.query(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum of values
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.out_proj(context)
        return output

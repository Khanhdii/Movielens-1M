import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        out, attn = self.attention(Q, K, V)
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.fc(out), attn

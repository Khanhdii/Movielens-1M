import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Tạo ma trận positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Đăng ký buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Lấy kích thước sequence length từ input
        seq_len = x.size(1)
        
        # Lấy positional encoding cho sequence length hiện tại
        pos_encoding = self.pe[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
        
        # Mở rộng pos_encoding để match batch size
        pos_encoding = pos_encoding.expand(x.size(0), -1, -1)  # [batch_size, seq_len, d_model]
        
        # Thêm positional encoding vào input
        return x + pos_encoding

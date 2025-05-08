import torch
import torch.nn as nn
from model.positional_encoding import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        
    def forward(self, src):
        # src shape: [batch_size, seq_len, d_model]
        # Thêm positional encoding
        src = self.pos_encoder(src)
        
        # Tạo mask cho padding
        src_key_padding_mask = (src == 0).all(dim=-1)
        
        # Chuyển qua transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        return output

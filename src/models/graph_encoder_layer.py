import torch
import torch.nn as nn
from layer_norm import LayerNorm

class GraphEncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, pre_layer_norm):
        super(GraphEncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_connections = nn.ModuleList([SublayerConnection(size, dropout, pre_layer_norm) for _ in range(2)])

    def forward(self, x, self_attn_bias, self_attn_mask, self_attn_padding_mask):
        # somehow define the attention input parameters in the parent class
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # added dropout after FFN
        return self.dropout(self.w_2(self.dropout(self.w_1(x).relu())))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        # Normalize before or after the self-attention and feedforward layers
        # see https://arxiv.org/pdf/2203.04810.pdf Section 2.1
        self.pre_layer_normalization = pre_layer_normalization # normalize before sublayer or after

    def forward(self, x, sublayer):
        if self.pre_layer_normalization:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return self.norm(x + self.dropout(sublayer(x)))
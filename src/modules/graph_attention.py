import torch
import torch.nn as nn
from einops import rearrange, repeat
import copy
import math

def make_deep_copy(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, attention_bias, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.einsum('bhij,bhkj->bhik', query, key) / math.sqrt(d_k)
    # we add the attention bias (information about the graph) to the scores before the softmax
    scores = scores + attention_bias 
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = make_deep_copy(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attention_bias, mask=None):

        if mask is not None:
            # The same mask is applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
          rearrange(lin(x), 'b n (h d) -> b h n d', h=self.h)
          for lin, x in zip(self.linears, (query, key, value))
        ]

        print("Attention V input:", value)

        x, self.attn = attention(
            query, key, value, attention_bias, mask=mask, dropout=self.dropout
        )

        x = rearrange(x, 'b h n d -> b n (h d)')
        print("Attention V output:", x)

        del query
        del key
        del value

        return self.linears[-1](x)

class GraphAttentionBias(nn.Module):
    def __init__(self):
        super(GraphAttentionBias, self).__init__()

    def forward(self, batch_data):
        return torch.zeros_like(batch_data)
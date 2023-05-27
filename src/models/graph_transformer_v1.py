import torch
import torch.nn as nn
import math
import copy
from einops import rearrange, reduce
from layer_norm import LayerNorm
from graph_embedding import GraphEmbeddingV1
from graph_attention_layer import GraphAttentionBias

torch.manual_seed(0)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GraphTransformerV1(nn.Module):
    def __init__(self, graph_embedding, graph_attention_bias, encoder_layer, N, d_model):
        super(GraphTransformerV1, self).__init__()
        self.graph_embedding = graph_embedding 
        self.graph_attention_bias = graph_attention_bias
        self.encoder_layers = clones(encoder_layer, N)
        self.d_model = d_model

        self.activation_fn = nn.GELU() # Maybe ReLu instead? or pass this as a parameter?
        self.layer_norm = LayerNorm(d_model)
        self.head_linear = nn.Linear(d_model, d_model)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, batch_data):
        x = self.graph_embedding(batch_data)
        attention_bias = self.graph_attention_bias(batch_data)

        # we first compute the hidden states for all nodes 
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_bias, mask)

        # extract the first node's hidden state for each graph in the batch 
        x = x[:, 0, :]

        x = self.layer_norm(self.activation_fn(self.head_linear(x)))
        return self.output_layer(x)


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
        self.linears = clones(nn.Linear(d_model, d_model), 4)
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

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def make_graph_transformer_model(
    src_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    transformer_layer = TransformerLayer(d_model, c(attn), c(ff), dropout)

    model = GraphTransformerV1(
        Embeddings(d_model, src_vocab),
        TransformerLayer(d_model, c(attn), c(ff), dropout),
        N=6,
        d_model=d_model
    )

    # Initialize parameters with Xavier uniform distribution 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

def inference_test():
    test_model = make_graph_transformer_model(11, 2)
    test_model.eval()
    batch = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    mask = torch.ones(1, 1, 10)
    attention_bias = torch.zeros(1, 1, 10)

    memory = test_model(batch, attention_bias, mask)

    print("Example Untrained Model Prediction:", memory)

# test the mask and the bias term are working
def inference_test_2():
    test_model = make_graph_transformer_model(3, 2)
    test_model.eval()
    batch = torch.LongTensor([[1, 1, 2]])
    mask = torch.tensor([[[True, True, True],
                          [True, True, True],
                          [True, True, True]]])
    attention_bias = torch.tensor([[[0, 0, 1e9],
                                    [0, 0, 1e9],
                                    [0, 0, 1e9]]])

    memory = test_model(batch, attention_bias, mask)

    print("Example Untrained Model Prediction:", memory)


if __name__ == "__main__":
    # inference_test()
    inference_test_2()
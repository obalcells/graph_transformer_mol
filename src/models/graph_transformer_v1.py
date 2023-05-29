import torch
import torch.nn as nn
import math
import logging
import copy
from einops import rearrange, reduce
from src.modules.graph_embedding import GraphEmbeddingV1
from src.modules.layer_norm import LayerNorm
from src.modules.graph_attention import MultiHeadedAttention, GraphAttentionBias
from src.modules.transformer_layer import TransformerLayer, PositionwiseFeedForward
from src.utils.parser import get_parser

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)

logger = logging.getLogger(__name__)
torch.manual_seed(0)

def make_deep_copy(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# we use this class to provide a wrapper for the graph transformer model
@register_model("graph_transformer_v1")
class GraphTransformer_V1_Wrapper(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

    @staticmethod
    def add_args(parser):
        parser.add_argument("--N", type=int, metavar="N", help="num layers")
        parser.add_argument("--d_model", type=int, metavar="N", help="hidden emb dimension")
        parser.add_argument("--max-nodes", type=int, help="max number of nodes in a graph")
        parser.add_argument("--h", type=int, help="num heads")
        parser.add_argument("--d_ff", type=int, help="hidden layer size in feedforward network")
        parser.add_argument("--dropout", type=float, help="dropout prob")

    @classmethod
    def build_model(cls, args, task):
        logger.info(args)

        model = GraphTransformer_V1(args)

        return cls(args, model) 

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphTransformer_V1(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.d_model = args.d_model
        self.N = args.N

        self_attention = MultiHeadedAttention(args.h, args.d_model)
        feed_forward = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)

        self.graph_embedding = GraphEmbeddingV1(args.h, args.d_model, 256, 100, 100)
        self.attention_bias = GraphAttentionBias()
        self.layers = make_deep_copy(TransformerLayer(
                                            args.d_model,
                                            self_attention,
                                            feed_forward,
                                            args.dropout,
                                            args.pre_layer_norm),
                                     args.N)

        del self_attention
        del feed_forward

        self.activation_fn = nn.GELU() # Maybe ReLu instead? or pass this as a parameter?
        self.layer_norm = LayerNorm(self.d_model)
        self.head_linear = nn.Linear(self.d_model, self.d_model)
        self.output_layer = nn.Linear(self.d_model, 1)

    def forward(self, batch_data, **unused):
        print("Calculating graph embedding...")
        x = self.graph_embedding(batch_data)
        print("Input tensor after embedding is:", x)
        attention_bias = self.attention_bias(batch_data)

        # we first compute the hidden states for all nodes 
        for layer in self.layers:
            x = layer(x, attention_bias)

        # extract the first node's hidden state for each graph in the batch 
        x = x[:, 0, :]

        x = self.layer_norm(self.activation_fn(self.head_linear(x)))
        # no dropout here
        return self.output_layer(x)

@register_model_architecture("graph_transformer_v1", "graph_transformer_v1")
def base_architecture(args):
    # args.dropout = getattr(args, "dropout", 0.1)
    args.dropout = 0.1
    # args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    # args.d_ff = getattr(args, "d_ff", 512)
    args.d_ff = 512
    # args.N = getattr(args, "encoder_layers", 4)
    args.N = 4
    # args.h = getattr(args, "encoder_attention_heads", 4)
    args.h = 4
    # args.max_nodes = getattr(args, "max_nodes", 256)
    args.max_nodes = 256
    # args.d_model = getattr(args, "d_model", 128)
    args.d_model = 128
    # args.pre_layer_norm = getattr(args, "pre_layer_norm", True)
    args.pre_layer_norm = True

def inference_test():
    parser = get_parser()
    args = parser.parse_args()
    # the non-specified arguments will be set to default base arch values
    base_architecture(args)

    test_model = GraphTransformer_V1(args)
    test_model.eval()
    batch = torch.LongTensor([[1, 2, 3, 2, 1]])
    mask = torch.ones(1, 1, 10)
    attention_bias = torch.zeros(1, 1, 10)

    memory = test_model(batch)

    print("Example Untrained Model Prediction:", memory)

# test the mask and the bias term are working
def inference_test_2():
    parser = get_parser()
    args = parser.parse_args()
    args.d_model = 4 
    args.h = 2
    # the non-specified arguments will be set to default base arch values
    base_architecture(args)

    test_model = GraphTransformer_V1(args)
    test_model.eval()
    batch = torch.LongTensor([[1, 1, 2]])
    mask = torch.tensor([[[True, True, True],
                          [True, True, True],
                          [True, True, True]]])
    attention_bias = torch.tensor([[[0, 0, 1e9],
                                    [0, 0, 1e9],
                                    [0, 0, 1e9]]])

    memory = test_model(batch)

    print("Example Untrained Model Prediction:", memory)


if __name__ == "__main__":
    # inference_test()
    inference_test_2()
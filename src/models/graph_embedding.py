import torch
import torch.nn as nn
import math

class GraphEmbeddingV1(nn.Module):
    def __init__(self, h, d_model, num_different_atoms, max_in_degree, max_out_degree):
        super(GraphEmbeddingV1, self).__init__()
        self.h = h
        self.d_model = d_model

        self.atom_type_embedding = nn.Embedding(num_different_atoms + 1, d_model, padding_idx=0)
        self.in_degree_embedding = nn.Embedding(max_in_degree + 1, d_model, padding_idx=0)
        self.out_degree_embedding = nn.Embedding(max_out_degree + 1, d_model, padding_idx=0)
        self.virtual_node_embedding = nn.Embedding(1, d_model) # this is the embedding for the virtual node in each graph

    def forward(self, batch_data):
        pass



from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph

# script taken from https://ogb.stanford.edu/docs/lsc/pcqm4mv2/#dataset

# smiles2graph takes a SMILES string as input and returns a graph object
# requires rdkit to be installed.
# You can write your own smiles2graph
graph_obj = smiles2graph('CC(NCC[C@H]([C@@H]1CCC(=CC1)C)C)C')

# convert each SMILES string into a molecular graph object by calling smiles2graph
# This takes a while (a few hours) for the first run
dataset = PygPCQM4Mv2Dataset(root = "/Users/oscarbalcells/Desktop/AI/task4", smiles2graph = smiles2graph)

print(dataset[1]) # (graph_obj, HOMO_LUMO gap)

only_smiles = PygPCQM4Mv2Dataset(root = "/Users/oscarbalcells/Desktop/AI/task4", only_smiles=True)

print(only_smiles[1])

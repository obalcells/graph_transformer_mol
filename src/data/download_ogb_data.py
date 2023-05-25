from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph

# smiles2graph takes a SMILES string as input and returns a graph object
# requires rdkit to be installed.
# You can write your own smiles2graph
graph_obj = smiles2graph('CC(NCC[C@H]([C@@H]1CCC(=CC1)C)C)C')

# convert each SMILES string into a molecular graph object by calling smiles2graph
# This takes a while (a few hours) for the first run
dataset = PCQM4Mv2Dataset(root = "/Users/oscarbalcells/Desktop/AI/task4/data", smiles2graph = smiles2graph)

print(dataset[0]) # (graph_obj, HOMO_LUMO gap)
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import networkx
import dgl

# valid datasets we have processed and can query
# please note that I've manually merged some into others so there's
# things which aren't implemented in all of them
IML = "IML"
CEP25000 = "CEP25000"
HOCPV15 = "HOCPV15"
ESM = "ESM"
QM9 = "QM9"
VALID_DATASET_IDS = [IML, CEP25000, HOCPV15, ESM, QM9]

class Normalizer():
    def __init__(self, data=None, target_mean=None, target_std=None):
        if target_mean is not None and target_std is not None:
            self.target_mean = target_mean
            self.target_std = target_std
        elif data is not None:
            self.target_mean = float(data.mean())
            self.target_std = float(data.std())
        else:
            assert False, "Incorrect normalizer initialization; must provide target data or mean and std of target data"

    def normalize(self, data):
        return (data - self.target_mean) / self.target_std

    def inv_normalize(self, data):
        return data * self.target_std + self.target_mean 


class GraphNNTorchDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_config):
        super().__init__()

        assert data_config["node_types"] is not None, "Node types have to be specified"

        self.data = data
        self.data_config = data_config
        self.normalizer = data_config["normalizer"]
        self.node_types = data_config["node_types"]
        self.column_x = data_config["column_x"]
        self.column_y = data_config["column_y"]

        self.node2idx = dict(self.node_types)
        self.max_node_idx = len(self.node2idx) - 1

        # this is to speed up the processing
        self.cached_processed_data = dict()

    @staticmethod
    def smiles2mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
        return mol

    # maps each atom to an index for example [Se, C, C_aromatic, Se] -> [0, 1, 2, 0]
    def mol2seq(self, mol):
        sequence = []
        for atom in mol.GetAtoms():
            node = (atom.GetSymbol(), atom.GetIsAromatic())
            try:
                idx = self.node2idx[node]
                if idx < -1 or idx > self.max_node_idx:
                    print("Found an unspecified atom in molecule")
                    idx = -1
            except KeyError as key_error:
                # in case we process a molecule with a new atom
                print("Found an unspecified atom in molecule")
                idx = -1
            sequence.append(idx)

        return sequence

    def process_smiles(self, smiles):
        cached_value = self.cached_processed_data.get(smiles)

        if cached_value is not None:
            sequence = cached_value[0]
            graph = cached_value[1]
            return sequence, graph

        mol = self.smiles2mol(smiles)

        sequence = self.mol2seq(mol)
        adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        graph_networkx_obj = networkx.convert_matrix.from_numpy_array(adj_matrix)
        graph = dgl.from_networkx(graph_networkx_obj)

        self.cached_processed_data[smiles] = (sequence, graph) # we cache this so we don't have to recompute it
        return sequence, graph

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        smiles = row[self.column_x] 
        # print(smiles)
        y = row[self.column_y]

        sequence, graph = self.process_smiles(smiles)

        if sequence is None and graph is None:
            return None

        atom_types = torch.as_tensor(sequence, dtype=torch.long)
        graph.ndata['atom_types'] = atom_types 

        y = torch.as_tensor(np.array(self.normalizer.normalize(y), dtype=np.float32))

        return [graph, y]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate(samples):
        # we use dgl's built-in batching method
        return [dgl.batch([item[0] for item in samples]), torch.as_tensor([item[1] for item in samples])]

def get_default_dataset_config(dataset_id, pre_config=None):
    assert dataset_id in VALID_DATASET_IDS


    if dataset_id == IML:
        iml_normalizer = Normalizer(target_mean=1.8889934551672973, target_std=0.37826160359173766)

        dataset_config = {
            "node_types": {('Se', True), ('S', True), ('Si', False), ('C', False), ('O', True), ('N', True), ('H', False), ('C', True)},
            "path": "./data/IML.csv",
            "max_num_atoms": 50,
            "target_mean": 1.8889934551672973,
            "target_std": 0.37826160359173766,
            "column_x": "smiles",
            "column_y": "homolumo",
            "split_column": "split",
            "normalizer": iml_normalizer 
        }
        # converting it into a dictionary
        dataset_config["node_types"] = {atom: idx for idx, atom in enumerate(dataset_config["node_types"])}
    elif dataset_id == CEP25000:
        cep_normalizer = Normalizer(target_mean=1.8927383253586847, target_std=0.3948395719406032)

        dataset_config = {
            "node_types": {('C', False): 0, ('C', True): 1, ('Se', True): 2, ('O', True): 3, ('N', True): 4, ('S', True): 5, ('H', False): 6, ('Si', False): 7}, 'wf_r1': {'atom_dict': {'C': 0, ('C', 'aromatic'): 1, ('Se', 'aromatic'): 2, ('O', 'aromatic'): 3, ('N', 'aromatic'): 4, ('S', 'aromatic'): 5, 'H': 6, 'Si': 7}, 'bond_dict': {'SINGLE': 0, 'DOUBLE': 1, 'AROMATIC': 2}, 'fragment_dict': {(0, ((0, 0), (0, 0), (6, 0), (6, 0))): 0, (0, ((0, 0), (0, 1), (6, 0))): 1, (0, ((0, 0), (0, 1), (1, 0))): 2, (1, ((0, 0), (1, 2), (4, 2))): 3, (1, ((1, 2), (1, 2), (6, 0))): 4, (1, ((1, 2), (1, 2), (2, 2))): 5, (2, ((1, 2), (1, 2))): 6, (1, ((1, 2), (1, 2), (3, 2))): 7, (3, ((1, 2), (1, 2))): 8, (1, ((1, 2), (3, 2), (6, 0))): 9, (1, ((1, 2), (1, 2), (1, 2))): 10, (1, ((1, 2), (1, 2), (4, 2))): 11, (4, ((1, 2), (5, 2))): 12, (5, ((4, 2), (4, 2))): 13, (1, ((1, 2), (4, 2), (6, 0))): 14, (4, ((1, 2), (1, 2))): 15, (6, ((0, 0),)): 16, (6, ((1, 0),)): 17, (1, ((0, 0), (1, 2), (1, 2))): 18, (1, ((0, 1), (1, 2), (1, 2))): 19, (0, ((1, 1), (6, 0), (7, 0))): 20, (7, ((0, 0), (0, 0), (6, 0), (6, 0))): 21, (6, ((7, 0),)): 22, (0, ((0, 0), (1, 1), (6, 0))): 23, (4, ((1, 2), (1, 2), (6, 0))): 24, (1, ((1, 2), (1, 2), (7, 0))): 25, (7, ((0, 0), (1, 0), (6, 0), (6, 0))): 26, (0, ((0, 0), (0, 1), (7, 0))): 27, (0, ((0, 1), (1, 0), (6, 0))): 28, (1, ((0, 0), (1, 2), (3, 2))): 29, (0, ((0, 0), (0, 0), (0, 1))): 30, (6, ((4, 0),)): 31, (0, ((0, 1), (6, 0), (7, 0))): 32, (1, ((0, 0), (1, 2), (2, 2))): 33, (1, ((1, 2), (2, 2), (6, 0))): 34, (1, ((1, 2), (5, 2), (6, 0))): 35, (5, ((1, 2), (1, 2))): 36, (1, ((1, 2), (1, 2), (5, 2))): 37, (1, ((0, 0), (1, 2), (5, 2))): 38, (0, ((0, 0), (1, 0), (6, 0), (6, 0))): 39, (0, ((0, 1), (1, 0), (7, 0))): 40, (1, ((4, 2), (5, 2), (6, 0))): 41, (1, ((0, 0), (4, 2), (4, 2))): 42, (1, ((4, 2), (4, 2), (6, 0))): 43, (1, ((0, 0), (4, 2), (5, 2))): 44, (1, ((1, 0), (1, 2), (4, 2))): 45, (1, ((1, 0), (1, 2), (3, 2))): 46, (1, ((1, 0), (1, 2), (5, 2))): 47, (1, ((1, 0), (1, 2), (1, 2))): 48, (1, ((1, 0), (1, 2), (2, 2))): 49, (1, ((1, 0), (4, 2), (4, 2))): 50, (1, ((1, 0), (4, 2), (5, 2))): 51, (0, ((4, 0), (6, 0), (6, 0), (6, 0))): 52, (4, ((0, 0), (1, 2), (1, 2))): 53, (0, ((1, 0), (1, 0), (6, 0), (6, 0))): 54, (7, ((1, 0), (1, 0), (6, 0), (6, 0))): 55}, 'edge_dict': {((0, 0), 0): 0, ((0, 6), 0): 1, ((0, 0), 1): 2, ((0, 1), 0): 3, ((1, 1), 2): 4, ((1, 4), 2): 5, ((1, 6), 0): 6, ((1, 2), 2): 7, ((1, 3), 2): 8, ((4, 5), 2): 9, ((0, 1), 1): 10, ((0, 7), 0): 11, ((6, 7), 0): 12, ((4, 6), 0): 13, ((1, 7), 0): 14, ((1, 5), 2): 15, ((1, 1), 0): 16, ((0, 4), 0): 17}},
            "path": "./data/CEPDB_25000.csv",
            "max_num_atoms": 53,
            "target_mean": 1.8927383253586847,
            "target_std": 0.3948395719406032,
            "column_x": "SMILES_str",
            "column_y": "e_gap_alpha",
            "split_column": "ml_phase",
            "normalizer": cep_normalizer 
        }
    elif dataset_id == HOCPV15:
        dataset_config = {
            "node_types": {('C', False): 0, ('C', True): 1, ('Se', True): 2, ('O', True): 3, ('N', True): 4, ('S', True): 5, ('H', False): 6, ('Si', False): 7, ('S', False): 8, ('O', False): 9, ('N', False): 10, ('F', False): 11}, 'wf_r1': {'atom_dict': {'C': 0, ('C', 'aromatic'): 1, ('Se', 'aromatic'): 2, ('O', 'aromatic'): 3, ('N', 'aromatic'): 4, ('S', 'aromatic'): 5, 'H': 6, 'Si': 7, 'S': 8, 'O': 9, 'N': 10, 'F': 11}, 'bond_dict': {'SINGLE': 0, 'DOUBLE': 1, 'AROMATIC': 2, 'TRIPLE': 3}, 'fragment_dict': {(0, ((0, 0), (0, 0), (6, 0), (6, 0))): 0, (0, ((0, 0), (0, 1), (6, 0))): 1, (0, ((0, 0), (0, 1), (1, 0))): 2, (1, ((0, 0), (1, 2), (4, 2))): 3, (1, ((1, 2), (1, 2), (6, 0))): 4, (1, ((1, 2), (1, 2), (2, 2))): 5, (2, ((1, 2), (1, 2))): 6, (1, ((1, 2), (1, 2), (3, 2))): 7, (3, ((1, 2), (1, 2))): 8, (1, ((1, 2), (3, 2), (6, 0))): 9, (1, ((1, 2), (1, 2), (1, 2))): 10, (1, ((1, 2), (1, 2), (4, 2))): 11, (4, ((1, 2), (5, 2))): 12, (5, ((4, 2), (4, 2))): 13, (1, ((1, 2), (4, 2), (6, 0))): 14, (4, ((1, 2), (1, 2))): 15, (6, ((0, 0),)): 16, (6, ((1, 0),)): 17, (1, ((0, 0), (1, 2), (1, 2))): 18, (1, ((0, 1), (1, 2), (1, 2))): 19, (0, ((1, 1), (6, 0), (7, 0))): 20, (7, ((0, 0), (0, 0), (6, 0), (6, 0))): 21, (6, ((7, 0),)): 22, (0, ((0, 0), (1, 1), (6, 0))): 23, (4, ((1, 2), (1, 2), (6, 0))): 24, (1, ((1, 2), (1, 2), (7, 0))): 25, (7, ((0, 0), (1, 0), (6, 0), (6, 0))): 26, (0, ((0, 0), (0, 1), (7, 0))): 27, (0, ((0, 1), (1, 0), (6, 0))): 28, (1, ((0, 0), (1, 2), (3, 2))): 29, (0, ((0, 0), (0, 0), (0, 1))): 30, (6, ((4, 0),)): 31, (0, ((0, 1), (6, 0), (7, 0))): 32, (1, ((0, 0), (1, 2), (2, 2))): 33, (1, ((1, 2), (2, 2), (6, 0))): 34, (1, ((1, 2), (5, 2), (6, 0))): 35, (5, ((1, 2), (1, 2))): 36, (1, ((1, 2), (1, 2), (5, 2))): 37, (1, ((0, 0), (1, 2), (5, 2))): 38, (0, ((0, 0), (1, 0), (6, 0), (6, 0))): 39, (0, ((0, 1), (1, 0), (7, 0))): 40, (1, ((4, 2), (5, 2), (6, 0))): 41, (1, ((0, 0), (4, 2), (4, 2))): 42, (1, ((4, 2), (4, 2), (6, 0))): 43, (1, ((0, 0), (4, 2), (5, 2))): 44, (1, ((1, 0), (1, 2), (4, 2))): 45, (1, ((1, 0), (1, 2), (3, 2))): 46, (1, ((1, 0), (1, 2), (5, 2))): 47, (1, ((1, 0), (1, 2), (1, 2))): 48, (1, ((1, 0), (1, 2), (2, 2))): 49, (1, ((1, 0), (4, 2), (4, 2))): 50, (1, ((1, 0), (4, 2), (5, 2))): 51, (0, ((4, 0), (6, 0), (6, 0), (6, 0))): 52, (4, ((0, 0), (1, 2), (1, 2))): 53, (0, ((1, 0), (1, 0), (6, 0), (6, 0))): 54, (7, ((1, 0), (1, 0), (6, 0), (6, 0))): 55, (0, ((1, 0), (6, 0), (6, 0), (6, 0))): 56, (1, ((1, 2), (5, 2), (8, 0))): 57, (8, ((0, 0), (1, 0), (9, 1), (9, 1))): 58, (0, ((6, 0), (6, 0), (6, 0), (8, 0))): 59, (9, ((8, 1),)): 60, (0, ((6, 0), (6, 0), (6, 0), (10, 0))): 61, (10, ((0, 0), (0, 0), (1, 0))): 62, (1, ((1, 2), (1, 2), (10, 0))): 63, (0, ((0, 0), (9, 1), (10, 0))): 64, (9, ((0, 1),)): 65, (7, ((0, 0), (0, 0), (0, 0), (0, 0))): 66, (0, ((6, 0), (6, 0), (6, 0), (7, 0))): 67, (1, ((1, 2), (1, 2), (11, 0))): 68, (11, ((1, 0),)): 69, (0, ((0, 3), (6, 0))): 70, (0, ((0, 3), (1, 0))): 71, (1, ((1, 2), (1, 2), (9, 0))): 72, (9, ((0, 0), (1, 0))): 73, (0, ((6, 0), (6, 0), (6, 0), (9, 0))): 74, (0, ((0, 0), (6, 0), (6, 0), (6, 0))): 75, (0, ((0, 0), (1, 0), (8, 1))): 76, (8, ((0, 1), (0, 1))): 77, (0, ((0, 0), (6, 0), (8, 1))): 78, (10, ((0, 0), (0, 0), (0, 0))): 79, (0, ((0, 1), (1, 0), (10, 0))): 80, (0, ((1, 0), (11, 0), (11, 0), (11, 0))): 81, (11, ((0, 0),)): 82, (0, ((0, 0), (0, 0), (1, 0), (1, 0))): 83, (10, ((1, 0), (1, 0), (1, 0))): 84, (0, ((0, 0), (10, 3))): 85, (10, ((0, 3),)): 86, (0, ((0, 0), (9, 0), (9, 1))): 87, (9, ((0, 0), (6, 0))): 88, (6, ((9, 0),)): 89, (0, ((0, 1), (6, 0), (6, 0))): 90, (0, ((0, 0), (0, 1), (8, 0))): 91, (8, ((0, 0), (0, 0))): 92, (0, ((8, 0), (8, 1), (10, 0))): 93, (8, ((0, 1),)): 94, (0, ((0, 0), (6, 0), (6, 0), (10, 0))): 95, (0, ((0, 0), (0, 0), (0, 0), (1, 0))): 96, (0, ((1, 0), (1, 0), (1, 0), (1, 0))): 97, (0, ((0, 0), (1, 0), (9, 1))): 98, (4, ((1, 0), (1, 2), (1, 2))): 99, (1, ((1, 2), (1, 2), (4, 0))): 100, (0, ((0, 1), (1, 0), (8, 0))): 101, (8, ((0, 0), (0, 0), (9, 1), (9, 1))): 102, (0, ((0, 3), (7, 0))): 103, (1, ((1, 2), (4, 2), (5, 2))): 104, (0, ((0, 0), (0, 0), (8, 1))): 105, (10, ((1, 0), (9, 0), (9, 1))): 106, (9, ((10, 1),)): 107, (9, ((10, 0),)): 108, (0, ((1, 0), (9, 0), (9, 1))): 109, (9, ((0, 0), (0, 0))): 110, (7, ((0, 0), (0, 0), (1, 0), (1, 0))): 111, (0, ((1, 0), (6, 0), (6, 0), (9, 0))): 112, (0, ((1, 0), (9, 1), (10, 0))): 113, (0, ((0, 0), (0, 0), (0, 0), (0, 0))): 114, (0, ((1, 0), (6, 0), (9, 1))): 115, (0, ((0, 1), (1, 0), (1, 0))): 116, (0, ((0, 0), (6, 0), (6, 0), (9, 0))): 117, (4, ((1, 2), (4, 2))): 118, (1, ((1, 2), (1, 2), (8, 1))): 119, (8, ((0, 1), (1, 1))): 120, (0, ((1, 0), (1, 0), (8, 1))): 121, (0, ((1, 0), (6, 0), (8, 1))): 122, (8, ((0, 0), (1, 0))): 123, (4, ((1, 2), (3, 2))): 124, (3, ((4, 2), (4, 2))): 125, (0, ((0, 0), (1, 0), (6, 0), (10, 0))): 126, (10, ((0, 0), (1, 1))): 127, (1, ((1, 2), (1, 2), (10, 1))): 128, (0, ((1, 0), (8, 0), (10, 1))): 129, (10, ((0, 0), (0, 1))): 130, (0, ((0, 0), (6, 0), (8, 0), (10, 0))): 131, (0, ((0, 1), (8, 0), (8, 0))): 132, (1, ((1, 2), (1, 2), (8, 0))): 133, (1, ((1, 2), (5, 2), (7, 0))): 134, (7, ((1, 0), (1, 0), (1, 0), (1, 0))): 135, (0, ((0, 0), (1, 0), (11, 0), (11, 0))): 136, (0, ((0, 0), (0, 0), (11, 0), (11, 0))): 137, (0, ((0, 0), (11, 0), (11, 0), (11, 0))): 138, (9, ((1, 0), (6, 0))): 139, (4, ((0, 0), (4, 2), (4, 2))): 140, (0, ((1, 0), (1, 0), (1, 0), (6, 0))): 141, (0, ((0, 0), (0, 0), (1, 0), (6, 0))): 142, (0, ((9, 0), (9, 1), (10, 0))): 143, (0, ((0, 0), (0, 0), (1, 0), (9, 0))): 144, (10, ((0, 0), (1, 0), (1, 0))): 145, (8, ((1, 0), (1, 0))): 146, (0, ((1, 0), (10, 3))): 147, (0, ((9, 1), (10, 0), (10, 0))): 148, (0, ((1, 0), (1, 0), (9, 1))): 149}, 'edge_dict': {((0, 0), 0): 0, ((0, 6), 0): 1, ((0, 0), 1): 2, ((0, 1), 0): 3, ((1, 1), 2): 4, ((1, 4), 2): 5, ((1, 6), 0): 6, ((1, 2), 2): 7, ((1, 3), 2): 8, ((4, 5), 2): 9, ((0, 1), 1): 10, ((0, 7), 0): 11, ((6, 7), 0): 12, ((4, 6), 0): 13, ((1, 7), 0): 14, ((1, 5), 2): 15, ((1, 1), 0): 16, ((0, 4), 0): 17, ((1, 8), 0): 18, ((0, 8), 0): 19, ((8, 9), 1): 20, ((0, 10), 0): 21, ((1, 10), 0): 22, ((0, 9), 1): 23, ((1, 11), 0): 24, ((0, 0), 3): 25, ((1, 9), 0): 26, ((0, 9), 0): 27, ((0, 8), 1): 28, ((0, 11), 0): 29, ((0, 10), 3): 30, ((6, 9), 0): 31, ((1, 4), 0): 32, ((9, 10), 1): 33, ((9, 10), 0): 34, ((4, 4), 2): 35, ((1, 8), 1): 36, ((3, 4), 2): 37, ((1, 10), 1): 38, ((0, 10), 1): 39}},
            "target_transform": lambda target: Normalizer.normalize(target, target_mean=1.8927383253586847, target_std=0.3948395719406032)
        }
    else:
        raise NotImplementedError()

    if pre_config is not None:
        # if pre_config doesn't have one key we put the default one
        for key in dataset_config.keys():
            if key not in pre_config.keys():
                pre_config[key] = dataset_config[key]
        return pre_config 
    else:
        return dataset_config

def get_default_normalizer(dataset_id):
    assert dataset_id in VALID_DATASET_IDS

    return get_default_dataset_config(dataset_id)["normalizer"]

def get_dataframes(dataset_id, splits=["train", "val", "test"]):
    assert dataset_id in VALID_DATASET_IDS

    path = get_default_dataset_config(dataset_id)["path"]
    df = pd.read_csv(path)

    if len(splits) == 0:
        return df
    else:
        list_df = []

        for split in splits:
            list_df.append(df[df["split"] == split])

        return tuple(list_df)

def get_data_loaders(dataset_id, list_of_splits=["train", "val", "test"], data_config=None, device=torch.device("cpu")):
    data_config = get_default_dataset_config(dataset_id, pre_config=data_config)

    # returns the whole dataframe
    dataset = get_dataframes(dataset_id, splits=[])

    split_column = data_config["split_column"]

    data_loaders = []
    for split_name in list_of_splits:
        gnn_dataset = GraphNNTorchDataset(dataset[dataset[split_column] == split_name], data_config)
        data_loader = DataLoader(gnn_dataset,
                                 batch_size=250,
                                 shuffle=True if split_name=="train" else False,
                                 collate_fn=GraphNNTorchDataset.collate,
                                 generator=torch.Generator(device))
        print(f"Loaded {split_name} data loader with {len(data_loader)} batches")
        data_loaders.append(data_loader)
        
    return tuple(data_loaders)



import pandas as pd
import numpy as np
from rdkit import Chem
import tarfile
import rdkit
from src.features.graph_preprocessing import preprocess_item, mol2graph, smiles2graph
from functools import lru_cache
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch
from ogb.utils.url import decide_download, download_url, extract_zip
import os
from multiprocessing import Pool
from tqdm import tqdm
from rdkit import Chem

def find_smiles_index(smiles_string, list_smiles):
    try:
        return list_smiles.index(smiles_string)
    except ValueError:
        return -1

def get_canonical_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:  # RDKit couldn't parse the SMILES
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)  # Use isomericSmiles=True to preserve stereochemistry

class IMLDataset(InMemoryDataset):
    def __init__(self, root="", smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = "/".join(list(root.split('/')[0:-1])) 
        # self.original_root = root
        self.smiles2graph = smiles2graph
        # self.folder = os.path.join(root, 'pcqm4m-v2')
        self.folder = root
        self.folder_name = "iml_task4"
        assert os.path.isdir(self.folder)
        self.version = 1

        super(IMLDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        # IML Data
        x_train = pd.read_csv(os.path.join(self.raw_dir, "train_features.csv.zip"), index_col="Id", compression='zip')
        y_train = np.load(os.path.join(self.raw_dir, "train_labels.npy"))
        x_test = pd.read_csv(os.path.join(self.raw_dir, "test_features.csv.zip"), index_col="Id", compression='zip')

        iml_smiles = list(pd.concat([x_train, x_test], axis=0)["smiles"])
        iml_homolumogap = np.concatenate([y_train, np.zeros(len(x_test))])

        print("SMILES strings", iml_smiles[:5])
        print("IML Homolumo gap", iml_homolumogap[:5])

        iml_train_smiles = iml_smiles[:len(x_train)]
        iml_test_smiles = iml_smiles[len(x_train):]
        assert len(iml_train_smiles) == 100 
        assert len(iml_test_smiles) == 10000

        # iml_smiles = [get_canonical_smiles(smiles) for smiles in iml_smiles]

        # OGB Data
        # data_df = pd.read_csv(os.path.join(self.raw_dir, 'data.csv.gz'))
        # ogb_smiles = list(data_df["smiles"])
        # ogb_smiles = [get_canonical_smiles(smiles) for smiles in ogb_smiles]
        # print("IML smiles:", iml_smiles[:5])
        # print("Ogb smiles:", ogb_smiles[:5])
        # homolumogap_list = data_df['homolumogap']
        # graph_pos_list = Chem.SDMolSupplier(os.path.join(self.original_root, 'pcqm4m-v2-train.sdf')) 

        data_list = []

        found = 0

        for i in range(len(iml_smiles)):
            # print("i = ", i)
            # j = find_smiles_index(iml_smiles[i], ogb_smiles) 

            # if j != -1:
            #     found += 1
        
            # if i <= 99:
            #     assert iml_smiles[i] == ogb_smiles[j]
            #     assert iml_homolumogap[i] == homolumogap_list[j]

            graph = smiles2graph(iml_smiles[i])
            homolumogap = iml_homolumogap[i]

            data = Data()
            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])
            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            # data.pos = torch.zeros(data.__num_nodes__, 3).to(torch.float32)

            data_list.append(data)

            # graph = mol2graph(graph_pos_list[j])

            # data = Data()
            # assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            # assert (len(graph['node_feat']) == graph['num_nodes'])
            # data.__num_nodes__ = int(graph['num_nodes'])
            # data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            # data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            # data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            # data.y = torch.Tensor([homolumogap])
            # data.pos = torch.from_numpy(graph['position']).to(torch.float32)

            # train_data_with_position_list.append(data)

        # print("Found:", found)

        # data_list = train_data_with_position_list + data_list[len(train_data_with_position_list):]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = {
            "train": torch.tensor(list(range(100))),
            "valid": torch.tensor(list(range(100))),
            "test-dev": torch.tensor(list(range(100, 1100))),
        }
        return split_dict

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)

if __name__ == "__main__":
    dataset = IMLDataset(root="/Users/oscarbalcells/Desktop/AI/task4/datasets/iml_task4/")
    print(len(dataset))
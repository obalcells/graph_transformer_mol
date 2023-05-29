from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from src.features.graph_preprocessing import preprocess_3d_graph, preprocess_2d_graph, mol2graph, smiles2graph
from functools import lru_cache
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import os
from ogb.utils.url import decide_download, download_url, extract_zip
import shutil
import torch

from multiprocessing import Pool

from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from rdkit import Chem
import tarfile
import rdkit

class MyPygPCQM4Mv2Dataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4Mv2Dataset, self).download()

    def process(self):
        super(MyPygPCQM4Mv2Dataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_2d_graph(item)


class PygPCQM4Mv2PosDataset(InMemoryDataset):
    def __init__(self, root='', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = os.path.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        self.pos_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'

        # check version and update if necessary
        if os.path.isdir(self.folder) and (not os.path.exists(os.path.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2PosDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

        if decide_download(self.pos_url):
            path = download_url(self.pos_url, self.original_root)
            tar = tarfile.open(path, 'r:gz')
            filenames = tar.getnames()
            for file in filenames:
                tar.extract(file, self.original_root)
            tar.close()
            os.unlink(path)
        else:
            print('Stop download')
            exit(-1)


    def process(self):
        data_df = pd.read_csv(os.path.join(self.raw_dir, 'data.csv.gz'))
        graph_pos_list = Chem.SDMolSupplier(os.path.join(self.original_root, 'pcqm4m-v2-train.sdf'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []
        with Pool(processes=1) as pool:
            iter = pool.imap(smiles2graph, smiles_list[:30])

            for i, graph in tqdm(enumerate(iter), total=len(homolumogap_list)):
                try:
                    data = Data()

                    homolumogap = homolumogap_list[i]

                    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                    assert (len(graph['node_feat']) == graph['num_nodes'])

                    data.__num_nodes__ = int(graph['num_nodes'])
                    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                    data.y = torch.Tensor([homolumogap])
                    data.pos = torch.zeros(data.__num_nodes__, 3).to(torch.float32)

                    data_list.append(data)
                    if len(data_list) == 30:
                        break
                except:
                    continue

        print('Extracting 3D positions from SDF files for Training Data...')
        train_data_with_position_list = []
        with Pool(processes=1) as pool:
            # iter = pool.imap(mol2graph, graph_pos_list)
            graph_list = []
            for i, mol in enumerate(graph_pos_list):
                if i == 30:
                    break 
                graph_list.append(mol2graph(mol))
            print("Graph list done", graph_list)
            print("Doing data list addition")

            # for i, graph in tqdm(enumerate(iter), total=len(graph_pos_list)):
            for i, graph in enumerate(graph_list):
                # try:
                data = Data()
                homolumogap = homolumogap_list[i]

                print("Step 1")

                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])

                print("Step 2")

                data.__num_nodes__ = int(graph['num_nodes'])
                print("Step 2 1")
                print("Edge index is ", graph['edge_index'])
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                print("Step 2 2")
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                print("Step 2 3")
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                print("Step 2 4")
                data.y = torch.Tensor([homolumogap])
                print("Step 2 5")
                data.pos = torch.from_numpy(graph['position']).to(torch.float32)

                print("Step 3")

                print("Appending graph", data)
                train_data_with_position_list.append(data)

                if len(train_data_with_position_list) == 30:
                    break
                # except:
                #     continue

        print("Train data with position list:", train_data_with_position_list)

        data_list = train_data_with_position_list + data_list[len(train_data_with_position_list):]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("Data list length is:", len(data_list))

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(os.path.join(self.root, 'split_dict.pt')))
        return split_dict


class MyPygPCQM4Mv2PosDataset(PygPCQM4Mv2PosDataset):
    def download(self):
        super(MyPygPCQM4Mv2PosDataset, self).download()

    def process(self):
        super(MyPygPCQM4Mv2PosDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_3d_graph(item)

if __name__ == "__main__":
    # dataset = PygPCQM4Mv2Dataset()
    dataset = MyPygPCQM4MPosv2Dataset()
    print(len(dataset))
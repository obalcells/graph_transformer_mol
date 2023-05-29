from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F

from fairseq.data import FairseqDataset, BaseWrapperDataset, data_utils

from ogb.lsc.pcqm4mv2 import PCQM4Mv2Evaluator
from src.data.ogb_dataset_wrappers import MyPygPCQM4Mv2Dataset, MyPygPCQM4Mv2PosDataset, SmallMyPygPCQM4Mv2PosDataset
from src.data.collater import collater_2d, collater_3d

# we call this class from fairseq to load the dataset
class PreprocessedData():
    def __init__(self, dataset_name="pcqm4m-v2-pos", dataset_path="/Users/oscarbalcells/Desktop/AI/task4/datasets/small-pcqm4m-v2-pos"):
        # only this dataset supported at the moment
        assert dataset_name in ["pcqm4m-v2", "pcqm4m-v2-pos", "iml-task4"], "Only pcqm4m-v2 and pcqm4m-v2-pos supported at the moment"

        self.dataset_name = dataset_name
        # root = "/".join(list(dataset_path.split('/')[0:-1])) 

        # we load a custom dataset class so that we can customise 
        # the __getitem__ method
        if dataset_name == "pcqm4m-v2":
            self.dataset = MyPygPCQM4Mv2Dataset(root=dataset_path)
        elif dataset_name == "pcqm4m-v2-pos":
            self.dataset = MyPygPCQM4Mv2PosDataset(root=dataset_path)
        elif dataset_name == "iml-task4":
            self.dataset = IMLDataset(root=dataset_path)

        self.setup()

    def setup(self, stage: str = None):
        # split_idx = self.dataset.get_idx_split()
        # we just use the first 1000 samples for now due to memory issues
        split_idx = {
            "train":torch.tensor(np.arange(0, 1000)),
            "valid":torch.tensor(np.arange(1000, 2000)),
            "test":torch.tensor(np.arange(2000, 3000)),
        }

        # the small version
        if len(self.dataset) < 3000:
            self.train_idx = torch.tensor([0, 1, 2, 3, 4, 5])
            self.valid_idx = torch.tensor([6, 7, 8])
            self.test_idx = torch.tensor([9, 10])
        else:
            self.train_idx = split_idx["train"]
            self.valid_idx = split_idx["valid"]
            self.test_idx = split_idx["test"]

        self.dataset_train = self.dataset.index_select(self.train_idx)
        self.dataset_test = self.dataset.index_select(self.test_idx)
        self.dataset_val = self.dataset.index_select(self.valid_idx)

        self.max_node = 256 
        self.multi_hop_max_dist = 5 
        self.spatial_pos_max = 1024 
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator()

class BatchedDataDataset(FairseqDataset):
    def __init__(self,
                 dataset,
                 dataset_version="2D",
                 max_node=256,
                 multi_hop_max_dist=5,
                 spatial_pos_max=1024):
        self.dataset = dataset

        assert dataset_version in ["2D", "3D"]
        self.dataset_version = dataset_version

        self.max_node = max_node
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_max = spatial_pos_max

    def __getitem__(self, index):
        return self.dataset[int(index)]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.dataset_version == "2D":
            return collater_2d(samples, 
                            max_node=self.max_node,
                            multi_hop_max_dist=self.multi_hop_max_dist,
                            spatial_pos_max=self.spatial_pos_max)
        elif self.dataset_version == "3D":
            return collater_3d(samples, 
                            max_node=self.max_node,
                            multi_hop_max_dist=self.multi_hop_max_dist,
                            spatial_pos_max=self.spatial_pos_max)
        else:
            raise NotImplementedError()

class CacheAllDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        return self.dataset[index]

    def collater(self, samples):
        return self.dataset.collater(samples)

class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, size, seed):
        super().__init__(dataset)
        self.size = size
        self.seed = seed
        self.set_epoch(1)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.size)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False


class TargetDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index].y

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return torch.stack(samples, dim=0)


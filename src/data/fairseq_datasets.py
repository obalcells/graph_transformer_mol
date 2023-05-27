from functools import lru_cache
import numpy as np
import torch
from torch.nn import functional as F

from fairseq.data import FairseqDataset, BaseWrapperDataset, data_utils

from ogb.lsc import PCQM4Mv2Evaluator
from .ogb_dataset_wrappers import MyPygPCQM4Mv2Dataset
from .collator import collator

# we call this class from fairseq to load the dataset
class PreprocessedData():
    def __init__(self, dataset_name="pcqm4m-v2", dataset_path="/Users/oscarbalcells/Desktop/AI/task4/datasets/pcqm4m-v2"):
        # only this dataset supported at the moment
        assert dataset_name == "pcqm4m-v2"
        self.dataset_name = dataset_name
        # we load a custom dataset class so that we can customise 
        # the __getitem__ method
        self.dataset = MyPygPCQM4Mv2Dataset(root=dataset_path)
        self.setup()

    def setup(self, stage: str = None):
        split_idx = self.dataset.get_idx_split()
        # we just use the first 1000 samples for now due to memory issues
        self.train_idx = split_idx["train"][:1000]
        self.valid_idx = split_idx["valid"][:1000]
        self.test_idx = split_idx["test"][:1000]

        self.dataset_train = self.dataset.index_select(self.train_idx)
        self.dataset_val = self.dataset.index_select(self.valid_idx)
        self.dataset_test = self.dataset.index_select(self.test_idx)

        self.max_node = 256 
        self.multi_hop_max_dist = 5 
        self.spatial_pos_max = 1024 
        self.loss_fn = F.l1_loss
        self.num_class = 1
        self.metric = 'mae'
        self.metric_mode = 'min'
        self.evaluator = PCQM4Mv2Evaluator()

class BatchedDataDataset(FairseqDataset):
    def __init__(self, dataset, dataset_version="2D"):
        self.dataset = dataset

        # the only supported version at the moment
        assert dataset_version == "2D"
        self.dataset_version = dataset_version

        self.max_node = dataset.max_node
        self.multi_hop_max_dist = dataset.multi_hop_max_dist
        self.spatial_pos_max = dataset.spatial_pos_max

    def __getitem__(self, index):
        item = self.dataset[int(index)]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if dataset_version == "3D":
            raise NotImplementedError()
        else:
            collater_fn = collator 
        return collater(samples, 
                        max_node=self.max_node,
                        multi_hop_max_distance=self.multi_hop_max_dist,
                        spatial_pos_max=self.spatial_pos_max)

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


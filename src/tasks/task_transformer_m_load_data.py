import logging
import math
import logging
import os, math

import contextlib
from dataclasses import dataclass, field
from typing import Optional
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from omegaconf import MISSING, II, open_dict, OmegaConf

import torch
import numpy as np
from fairseq.data import (
    ConcatSentencesDataset,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    RollDataset,
    SortDataset,
    StripTokenDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq import models

from src.data.fairseq_datasets import PreprocessedData
from src.models.graph_transformer_v1 import GraphTransformer_V1
from src.data.fairseq_datasets import (
    PreprocessedData,
    BatchedDataDataset,
    CacheAllDataset,
    EpochShuffleDataset,
    TargetDataset
)
from src.criterions.graph_regression import GraphRegressionL1Loss, GraphRegressionCriterion 
from src.models.transformer_m import TransformerMModel

logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])

@dataclass
class TransformerMPredictionConfig(FairseqDataclass):
    # data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    data_path: str = field(
        default="",
        metadata={
            "help": "path to data file"
        },
    )

    noise_scale: float = field(
        default=0.01,
        metadata={
            "help": "noise scale"
        },
    )

    sandwich_ln: bool = field(
        default=False,
        metadata={"help": "apply layernorm via sandwich form"},
    )

    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )
    no_shuffle: bool = field(
        default=False,
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    add_prev_output_tokens: bool = field(
        default=False,
        metadata={
            "help": "add prev_output_tokens to sample, used for encoder-decoder arch"
        },
    )
    max_positions: int = field(
        default=512,
        metadata={"help": "max tokens per example"},
    )

    dataset_name: str = field(
        default="pcqm4m-v2",
        metadata={"help": "name of the dataset"},
    )

    num_atoms: int = field(
        default=512 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=512 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    def __len__(self):
        return 0

@register_task("transformer_m_inference")
class TransformerMPredictionTask(FairseqTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dm = PreprocessedData(dataset_name=self.cfg.dataset_name, dataset_path=self.cfg.data_path)
        print("Length of dataset train: ", len(self.dm.dataset_train))
        print("Length of dataset valid: ", len(self.dm.dataset_train))
        print("Length of dataset test: ", len(self.dm.dataset_test))

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test-dev, test-challenge)."""

        assert split in ["train", "valid", "test-dev"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test-dev":
            batched_data = self.dm.dataset_test

        batched_dataset = BatchedDataDataset(batched_data,
            dataset_version="3D",
            max_node=self.dm.max_node,
            multi_hop_max_dist=self.dm.multi_hop_max_dist,
            spatial_pos_max=self.dm.spatial_pos_max
        )

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset({
            "nsamples": NumSamplesDataset(),
            "net_input": {
                "batched_data": batched_dataset
            },
            "target": target
        }, sizes=np.array([1] * len(batched_dataset))) # FIXME: workaroud, make all samples have a valid size

        if split == "train":
            dataset = EpochShuffleDataset(dataset, size=len(batched_data), seed=0)

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        return models.build_model(cfg, self)

    def build_criterion(self, cfg):
        from fairseq import criterions

        return criterions.build_criterion(cfg, self)

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary

if __name__ == "__main__":
    cfg = TransformerMPredictionConfig()
    cfg.arch = "transformer_m_base"
    # cfg.criterion = "l1_loss"
    cfg.criterion = "m_graph_regression"
    cfg.dataset_name = "pcqm4m-v2-pos"
    cfg.data_path = "/users/oscarbalcells/Desktop/AI/task4/datasets/pcqm4m-v2"
    task = TransformerMPredictionTask(cfg)

    model = task.build_model(cfg)
    criterion = task.build_criterion(cfg)

    task.load_dataset('train')
    task.load_dataset('valid')

    batch_itr = task.get_batch_iterator(
        task.dataset("train"), max_tokens=4096
    )
    batch_itr = batch_itr.next_epoch_itr()

    for batch in batch_itr:
        loss, sample_size, logging_output = criterion(model, batch) 
        print(f"Loss {loss / sample_size}")
        loss.backward()

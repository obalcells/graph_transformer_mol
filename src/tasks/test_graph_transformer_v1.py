import logging
import math
import os


import torch
import numpy as np
from dataclasses import dataclass
import contextlib
from typing import Optional
from omegaconf import MISSING, II, open_dict, OmegaConf

from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.dataclass import ChoiceEnum
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
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.dataclass import ChoiceEnum
from fairseq.optim.amp_optimizer import AMPOptimizer

from ..data.dataset import (
    BatchedDataDataset,
    TargetDataset,
)

logger = logging.getLogger(__name__)
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])

@register_task("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionTask(FairseqTask):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dm = PCQPreprocessedData(dataset_name=self.cfg.dataset_name, dataset_path=self.cfg.data_path)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test

        batched_data = BatchedDataDataset(batched_data,
            dataset_version="2D" if self.cfg.dataset_name == 'PCQM4M-LSC-V2' else "3D",
            max_node=self.dm.max_node,
            multi_hop_max_dist=self.dm.multi_hop_max_dist,
            spatial_pos_max=self.dm.spatial_pos_max
        )

        target = TargetDataset(batched_data)

        dataset = NestedDictionaryDataset({
            "nsamples": NumSamplesDataset(),
            "net_input": {
                "batched_data": batched_data
            },
            "target": target
        }, sizes=np.array([1] * len(batched_data))) # FIXME: workaroud, make all samples have a valid size

        if split == "train":
            dataset = EpochShuffleDataset(dataset, size=len(batched_data), seed=self.cfg.seed)

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        return model

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
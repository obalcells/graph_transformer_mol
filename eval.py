import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.tasks.task_transformer_m_inference import TransformerMPredictionTask, TransformerMPredictionConfig
from src.models.transformer_m import TransformerMModel

import sys
from os import path

def rmse(predictions, targets):
    return 

def eval():
    cfg = TransformerMPredictionConfig()
    cfg.seed = 1
    cfg.arch = "transformer_m_base"
    cfg.dataset_name = "pcqm4m-v2-pos"
    cfg.data_path = "/users/oscarbalcells/Desktop/AI/task4/datasets/pcqm4m-v2-pos"
    cfg.encoder_layers = 12
    cfg.encoder_embed_dim = 768
    cfg.encoder_ffn_embed_dim = 768
    cfg.encoder_attention_heads = 32
    cfg.num_3d_bias_kernel = 128
    cfg.add_3d = True
    checkpoint_path = "/Users/oscarbalcells/Desktop/AI/task4/models/L12.pt"
    metric = "rmse"
    split = "train"

    np.random.seed(cfg.seed)
    utils.set_torch_seed(cfg.seed)

    task = TransformerMPredictionTask(cfg)
    model = task.build_model(cfg)

    # load checkpoint
    model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(
        model_state, strict=True
    )
    del model_state

    # model.to(torch.cuda.current_device())
    # load dataset
    task.load_dataset(split)

    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        disable_iterator_cache=False,
    )

    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )

    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(itr):
            y = model(**sample["net_input"])[0][:, 0, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])

    print(y_pred)
    print(y_true)

    # save predictions
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)

    if metric == "rmse":
        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
        print(f"RMSE: {rmse}")
    elif metric == "mae":
        mae = np.mean(np.abs(y_true.numpy() - y_pred.numpy()))
        print(f"mae: {mae}")
        return mae
    else:
        raise ValueError(f"Unsupported metric {args.metric}")

if __name__ == '__main__':
    eval()

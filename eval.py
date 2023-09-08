import torch
import numpy as np
import pandas as pd
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from src.features.graph_preprocessing import smiles2graph, preprocess_item
from src.data.collater import collater_2d
from torch_geometric.data import Data
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.tasks.prediction_graph import GraphPredictionConfig, GraphPredictionTask 
from src.models.transformer_m import TransformerMModel
from src.models.graphormer import GraphormerModel

import sys
from os import path
from src.data.iml_test_dataset import get_canonical_smiles

def get_raw_data():
    raw_dir = "/Users/oscarbalcells/Desktop/AI/task4/datasets/iml-task4/raw"
    x_train = pd.read_csv(os.path.join(raw_dir, "train_features.csv.zip"), index_col="Id", compression='zip')
    y_train = np.load(os.path.join(raw_dir, "train_labels.npy"))
    x_test = pd.read_csv(os.path.join(raw_dir, "test_features.csv.zip"), index_col="Id", compression='zip')
    iml_smiles = list(pd.concat([x_train, x_test], axis=0)["smiles"])[:3]
    iml_homolumogap = np.concatenate([y_train, np.zeros(len(x_test))])[:3]

    print("IML smiles :3", iml_smiles)
    print("IML gaps   :3", iml_homolumogap)

    # indices = [993]
    data_df = pd.read_csv("/Users/oscarbalcells/Desktop/AI/task4/datasets/pcqm4m-v2/raw/data.csv.gz")
    ogb_smiles = data_df['smiles'][:10]
    ogb_homolumogap = np.array(data_df['homolumogap'][:10])

    return iml_smiles, iml_homolumogap, ogb_smiles, ogb_homolumogap

def build_batch_from_smiles(smiles_list, homolumogap_list=None):
    data_list = []

    for i, smiles in enumerate(smiles_list):
        smiles = get_canonical_smiles(smiles)
        graph = smiles2graph(smiles)
        data = Data()
        assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert (len(graph['node_feat']) == graph['num_nodes'])
        data.idx = 0 
        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        if homolumogap_list is not None:
            data.y = torch.Tensor([homolumogap_list[i]])
        else:
            data.y = torch.Tensor([0.0])
        # data.pos = torch.zeros(data.__num_nodes__, 3).to(torch.float32)

        data_list.append(preprocess_item(data))

    batch = collater_2d(data_list,
                        max_node=256,
                        multi_hop_max_dist=5,
                        spatial_pos_max=1024)

    return batch

def visual_batch():
    indices = [993, 859, 298]

    data_df = pd.read_csv("/Users/oscarbalcells/Desktop/AI/task4/datasets/pcqm4m-v2/raw/data.csv.gz")
    ogb_smiles = data_df['smiles'].iloc[indices]
    ogb_homolumogap = np.array(data_df['homolumogap'].iloc[indices])

    sample = build_batch_from_smiles(ogb_smiles, ogb_homolumogap)

    for el in sample.keys():
        if el != "idx":
            print(f"{el} is {sample[el][:3]}")
            print(f"{el} has shape {sample[el][:3].shape}")



def manual_eval():
    cfg = GraphPredictionConfig()
    cfg.seed = 1
    cfg.arch = "graphormer_base"
    cfg.dataset_name = "pcqm4m-v2"
    # cfg.data_path = "/users/oscarbalcells/Desktop/AI/task4/datasets/iml-task4"
    cfg.data_path = "/users/oscarbalcells/Desktop/AI/task4/datasets/pcqm4m-v2"
    cfg.max_nodes = 512
    cfg.encoder_layers = 12
    cfg.num_classes = 1
    checkpoint_path = "/Users/oscarbalcells/Desktop/AI/task4/models/checkpoint_best_pcqm4mv2.pt"
    metric = "rmse"
    split = "valid"

    task = GraphPredictionTask(cfg)
    model = task.build_model(cfg)

    # load checkpoint
    model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(
        model_state, strict=True
    )
    del model_state

    # iml_smiles, iml_homolumogap, ogb_smiles, ogb_homolumogap = get_raw_data()
    iml_smiles = ["[SiH2]1C=Cc2sc3c([nH]c4cc(oc34)-c3nccc4nsnc34)c12"] # CC(NCC[C@H]([C@@H]1CCC(=CC1)C)C)C"]
    iml_homolumogap = np.array([0.0])

    iml_batch = build_batch_from_smiles(iml_smiles)
    # ogb_batch = build_batch_from_smiles(ogb_smiles, ogb_homolumogap)

    # for key in ogb_batch:
    #     if key != "idx":
    #         print(f"{key}: {ogb_batch[key][:3]}")
    #         print(f"{key} has shape {ogb_batch[key][:3].shape}")
    # not indegree, not spatial pos, 
    # print(ogb_batch["attn_edge_type"][0][:13, :13, :])
    # print(ogb_batch["attn_edge_type"][:3].shape)

    # print("Manual batch", iml_batch, file=open("manual_batch.txt", "w"))
    # torch.save(ogb_batch, "manual_batch.pt")
    
    with torch.no_grad():
        model.eval()
        iml_y = model(iml_batch)[:, 0, :].reshape(-1).cpu().numpy()
        # ogb_y = model(ogb_batch)[:, 0, :].reshape(-1).cpu().numpy()

    print("IML Y", iml_y)
    print("IML Gap", iml_homolumogap)
    # print("OGB Y", ogb_y)
    # print("OGB Gap", ogb_homolumogap)
    iml_rmse = np.sqrt(((iml_y - iml_homolumogap) ** 2).mean()) 
    # ogb_rmse = np.sqrt(((ogb_y - ogb_homolumogap) ** 2).mean())

    print("IML RMSE {0}".format(iml_rmse))
    # print("OGB RMSE {0}".format(ogb_rmse))

def eval():
    cfg = GraphPredictionConfig()
    cfg.seed = 1
    cfg.arch = "graphormer_base"
    cfg.dataset_name = "pcqm4m-v2"
    # cfg.data_path = "/users/oscarbalcells/Desktop/AI/task4/datasets/iml-task4"
    cfg.data_path = "/users/oscarbalcells/Desktop/AI/task4/datasets/pcqm4m-v2"
    cfg.max_nodes = 512
    cfg.encoder_layers = 12
    cfg.num_classes = 1
    checkpoint_path = "/Users/oscarbalcells/Desktop/AI/task4/models/checkpoint_best_pcqm4mv2.pt"
    metric = "rmse"
    split = "train"

    np.random.seed(cfg.seed)
    utils.set_torch_seed(cfg.seed)

    task = GraphPredictionTask(cfg)
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
            # for el in sample["net_input"]["batched_data"].keys():
            #     if el != "idx":
            #         print(f"{el} is {sample['net_input']['batched_data'][el][:3]}")
            #         print(f"{el} has shape {sample['net_input']['batched_data'][el][:3].shape}")
            # print("In degree:", sample["net_input"]["batched_data"]["in_degree"][:3])
            # print("Spatial pos:", sample["net_input"]["batched_data"]["attn_edge_type"][:3])
            # print("Spatial pos shape", sample["net_input"]["batched_data"]["attn_edge_type"][:3].shape)
            # print("Attn edge type pos:", sample["net_input"]["batched_data"]["attn_edge_type"][0][:, 13:, :])
            # print("Idx:", sample["net_input"]["batched_data"]["idx"][:3])
            first_element = sample["net_input"]["batched_data"]

            # for key in first_element.keys():
            #     if key != "idx":
            #         first_element[key] = first_element[key][:1]

            # torch.save(first_element, "fairseq_batch.pt")
            # print("Fairseq batch:", first_element, file=open("fairseq_batch.txt", "w"))

            y = model(**sample["net_input"])[:, 0, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])

            break

    print("Predictions:", y_pred[:10])
    print("True values:", y_true[:10])

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

import sys

if __name__ == '__main__':
    if sys.argv[1] == "eval":
        eval()
    elif sys.argv[1] == "manual_eval":
        manual_eval()
    elif sys.argv[1] == "visual_batch":
        visual_batch()
    else:
        assert False

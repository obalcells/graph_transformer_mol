import pandas as pd
import numpy as np
import os

def load_data(root_dir="", load_as_np=True):
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
	"""
	
    if load_as_np:
        x_pretrain = pd.read_csv(os.path.join(root_dir, "data/raw/pretrain_features.csv.zip"), index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
        y_pretrain = pd.read_csv(os.path.join(root_dir, "data/raw/pretrain_labels.csv.zip"), index_col="Id", compression='zip').to_numpy().squeeze(-1)
        x_train = pd.read_csv(os.path.join(root_dir, "data/raw/train_features.csv.zip"), index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
        y_train = pd.read_csv(os.path.join(root_dir, "data/raw/train_labels.csv.zip"), index_col="Id", compression='zip').to_numpy().squeeze(-1)
        x_test = pd.read_csv(os.path.join(root_dir, "data/raw/test_features.csv.zip"), index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    else:
        x_pretrain = pd.read_csv(os.path.join(root_dir, "data/raw/pretrain_features.csv.zip"), index_col="Id", compression='zip')
        y_pretrain = pd.read_csv(os.path.join(root_dir, "data/raw/pretrain_labels.csv.zip"), index_col="Id", compression='zip').to_numpy().squeeze(-1)
        x_train = pd.read_csv(os.path.join(root_dir, "data/raw/train_features.csv.zip"), index_col="Id", compression='zip')
        y_train = pd.read_csv(os.path.join(root_dir, "data/raw/train_labels.csv.zip"), index_col="Id", compression='zip').to_numpy().squeeze(-1)
        x_test = pd.read_csv(os.path.join(root_dir, "data/raw/test_features.csv.zip"), index_col="Id", compression='zip')

    return x_pretrain, y_pretrain, x_train, y_train, x_test
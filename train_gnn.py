import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import IML, CEP25000
from src.data import *
from src.models.graph_nn import GraphNN

device = torch.device("cpu")

def train_gnn_model(model, label, data_config):
    # training params
    epochs = 500,
    patience = 50,
    min_delta = 0.0,
    metric = "special iml rmse" # <- we keep track of it with pytorch-lightning
    direction = "min"
    test_name = label

    train_dl, val_dl, _ = get_data_loaders(data_config["id"],
                                           list_of_splits=["train", "val", "test"],
                                           data_config=data_config,
                                           device=device)

    print("Data loaders loaded")

    gpus = 1 if torch.cuda.is_available() else 0

    callbacks = []

    early_stopping_callback = EarlyStopping(monitor=metric,
                                            patience=patience,
                                            min_delta=min_delta,
                                            verbose=True,
                                            mode=direction)
    callbacks.append(early_stopping_callback)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="/model_checkpoints/", filename=label, monitor=metric, save_last=True, save_top_k=1)
    callbacks.append(checkpoint_callback)

    csv_logger = pl.loggers.CSVLogger(save_dir="./logs/", name=label)

    trainer_params = {
        'max_epochs': epochs,
        'logger': csv_logger,
        'callbacks': callbacks,
        'profiler': 'simple',
        'log_every_n_steps': 5 
    }

    if gpus > 0:
        trainer_params.update({
            "accelerator": "gpu",
            "devices": 1
        })

    trainer = pl.Trainer(**trainer_params)

    trainer.fit(model, train_dl, val_dl)

    val_error = early_stopping_callback.best_score.item()
    print(f"Val error {val_error}")

if __name__ == "__main__":

    model_params = {
        "emb_dim": 256,
        "conv_layers": 6*[256],
        "num_atom_types": 8,
        "linear_layers": [144, 144, 1],
        "padding_idx": 0,
        "dropout": 1e-1,
        "optimizer": {
            'name': 'Adam',
            'lr': 8e-4,
            'weight_decay': 2e-5,
        }, 
    }

    iml_normalizer = get_default_normalizer(IML)

    data_config = {
        "id": CEP25000,
        "src": "./data/processed/CEPDB_25000.csv",
        'split_column': "ml_phase",
        "normalizer": iml_normalizer # we train using the IML normalizer
    }

    model = GraphNN(**model_params)

    model.special_batches = [] # [IML train batch, other batches you might want to test, ...]
    # manually assign the first batch of IML training data
    # For this you can edit the function `eval_model` to return the first batch instead of computing anything
    # Important: The first batch of IML training data is used as validation loss for early-stopping the training

    # checkpoint_path = "final_model.ckpt"
    # if device.type == "cpu":
    #     checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # else:
    #     checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['state_dict'], strict=True)

    train_gnn_model(model, "last_training_run", data_config=data_config)

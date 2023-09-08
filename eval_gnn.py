import torch
from src.data import IML
from src.data import *
from src.models.graph_nn import GraphNN

device = torch.device("cpu")

def eval_gnn_model(model, data_config=None):
    model.eval()

    train_dl, val_dl, test_dl = get_data_loaders(data_config["id"],
                                          list_of_splits=["train", "val", "test"],
                                          data_config=data_config,
                                          device=device)

    normalizer = data_config["normalizer"]

    with torch.no_grad():
        for split, dl in zip(["test"], [test_dl]):
            batch_number = 0
            y_hat_all = []
            y_all = []
    
            for x_batch, y_batch in dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                batch_number += 1

                output_batch = model(x_batch)

                y = normalizer.inv_normalize(y_batch)
                y_hat = normalizer.inv_normalize(output_batch)

                rmse_batch = torch.sqrt(((y - y_hat) ** 2).mean())
                print(batch_number, '/ len', len(dl), '/ batch rmse', rmse_batch.item())

                y_all.extend(y.detach().cpu())
                y_hat_all.extend(y_hat.detach().cpu())

            if split == "test":
                np.save("./predictions.npy", y_hat_all)

            y_all = torch.Tensor(y_all)
            y_hat_all = torch.Tensor(y_hat_all)

            rmse = np.sqrt(((y_all - y_hat_all) ** 2).mean())
            mae = np.abs(y_all - y_hat_all).mean()
    
            y = y_all.cpu().numpy()
            y_pred = y_hat_all.cpu().numpy()
    
            print(f"For split {split} rmse is {rmse} and mae is {mae}")

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

    iml_data_config = {
        'id': IML,
        "src": "./data/IML.csv",
        "x_column": "smiles",
        "y_column": "homolumo",
        'split_column': "split",
        "normalizer": iml_normalizer,
        "node_types": {('C', False): 0, ('C', True): 1, ('Se', True): 2, ('O', True): 3, ('N', True): 4, ('S', True): 5, ('H', False): 6, ('Si', False): 7}
    }

    model = GraphNN(**model_params)

    checkpoint_path = "model.ckpt"
    if device.type == "cpu":
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"], strict=True)

    eval_gnn_model(model, data_config=iml_data_config)

    # storing results.csv file
    predictions = np.load("./predictions.npy")
    print("First 10 predictions:", predictions[:10])
    x_test_index = pd.read_csv('./data/test_features.csv.zip', index_col="Id", compression='zip').drop("smiles", axis=1).index
    predictions_df = pd.DataFrame({"y": predictions}, index=x_test_index)
    predictions_df.to_csv("results.csv")

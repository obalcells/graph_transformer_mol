import numpy as np
import pytorch_lightning as pl
import dgl
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

class GraphNNPytorchLightningWrapper(pl.LightningModule):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer

    def configure_optimizers(self):
        opt_params = self.optimizer.copy()
        name = opt_params.pop('name')
        optimizer_instance = getattr(torch.optim, name)(self.parameters(), **opt_params)
        return optimizer_instance

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log('phase', 'train')
        self.log('loss', loss)
        self.log('count', len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # self.validation_step_outputs = y_hat
        loss = torch.nn.MSELoss()(y_hat, y)
        self.log("val_rmse", loss)
        if batch_idx % 5 == 0:
            print("val loss ", torch.sqrt(loss).item())
        return y, y_hat

    # def validation_epoch_end(self, outputs):
    def on_validation_epoch_end(self, trainer, pl_module):
        # there's a weird thing with pl with these callbacks
        # it's too hard to save up the validation predictions somewhere
        # so I just have a list of "special" batches that I run after every validation epoch
        rmse_scores = self.test_special_batch()
        iml_rmse, val_rmse = rmse_scores[0], rmse_scores[1] 
        self.log("special iml rmse", iml_rmse) # <- this is the metric we track for the early stopping
        self.log("special val rmse", val_rmse)
        outputs = None #self.validation_step_outputs
        return

        # return super().on_validation_epoch_end(outputs)
        # return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y, y_hat

class GraphNN(GraphNNPytorchLightningWrapper):
    def __init__(self, emb_dim, num_atom_types, optimizer, conv_layers=6*[128], linear_layers=[100, 1], padding_idx=None, dropout=0.0):
        super().__init__(optimizer)
        self.padding_idx = padding_idx
        num_atom_types += int(padding_idx is not None)
        self.emb = nn.Embedding(num_atom_types, emb_dim, padding_idx=padding_idx)

        self.conv_layers = nn.ModuleList([GraphNNLayer(input_dim, output_dim, F.relu) for input_dim, output_dim in zip([emb_dim]+conv_layers, conv_layers)])

        linear_layers = [conv_layers[-1]] + linear_layers 
        seq_linear_layers = []
        for i in range(len(linear_layers) - 2):
            seq_linear_layers.append(nn.Linear(linear_layers[i], linear_layers[i+1]))
            seq_linear_layers.append(nn.ReLU())
            seq_linear_layers.append(nn.Dropout(dropout))

        # the last layer doesn't have RELU or dropout
        seq_linear_layers.append(nn.Linear(linear_layers[len(linear_layers)-2], linear_layers[len(linear_layers)-1]))

        self.linear_layers = nn.Sequential(*seq_linear_layers)
        
        self.one = torch.Tensor([1]).long()

    def forward(self, batch):
        atom_types = batch.ndata["atom_types"]

        if self.padding_idx is not None:
            atom_types += self.one.to(self.device).expand(atom_types.size())

        h = self.emb(atom_types)
        for layer in self.conv_layers:
            h = layer(batch, h)

        batch.ndata['h'] = h
        mean = dgl.mean_nodes(batch, 'h')

        y = rearrange(self.linear_layers(mean), "b 1 -> b")

        return y

    # we set this special batch manually
    def test_special_batch(self):
        rmse_scores = []
        for name, x, y in self.special_batches: 
            y_hat = self(x).detach()
            rmse_batch = torch.sqrt(torch.nn.MSELoss()(y_hat, y))
            rmse_scores.append(rmse_batch)
            print(f"RMSE batch for {name} is {rmse_batch.item()}")
        return rmse_scores


class GraphNNLayer(pl.LightningModule):
    def __init__(self, input_dim, output_dim, activation_fct):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation_fct = activation_fct
        self.msg = dgl.function.copy_src(src='h', out='m')
        self.reduce = dgl.function.mean(msg='m', out='h')

    def forward(self, g, h_input_features):
        with g.local_scope():
            g.ndata['h'] = h_input_features
            g.update_all(self.msg, self.reduce)
            h = g.ndata['h']
            h = self.linear(h)
            h += h_input_features
            return self.activation_fct(h)


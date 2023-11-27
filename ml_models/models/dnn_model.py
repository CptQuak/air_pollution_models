import torch
from torch import nn
import lightning.pytorch as pl
from torch.nn import functional as F
from torch import optim
import pandas as pd

hl=46
hl2=32


class DNN_model(nn.Module):
    def __init__(self, seq_gen, hidden_list, dropout=.3):
        super().__init__()
        self.seq_gen = seq_gen
        self.hidden_list = hidden_list
        self.input_shape = int(len(self.seq_gen.input_columns_idx) * self.seq_gen.input_width, )
        self.output_shape = int(len(self.seq_gen.output_columns_idx) * self.seq_gen.output_width)

        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.input_shape, self.hidden_list[0]))
        for i in range(1, len(self.hidden_list)):
            self.fc_layers.append(nn.Linear(self.hidden_list[i-1], self.hidden_list[i]))
        self.fc_layers.append(nn.Linear(self.hidden_list[-1], self.output_shape))
        self.dropout = nn.Dropout(dropout) 


    def forward(self, X):
        X = X.swapaxes(1, 2).reshape(-1, self.input_shape)
        for fc_layer in self.fc_layers[:-1]:
            X = self.dropout(torch.relu(fc_layer(X)))
        X = self.fc_layers[-1](X)

        return X.reshape(-1, self.seq_gen.output_width, len(self.seq_gen.output_columns_idx))
    
    ## LIGHTNING TRAINING UTILITY FUNCTIONS
    def configure_optimizers(self, lr=1e-3, weight_decay=0.0003):
        optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay)
        return optimizer

    def step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('training_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        val_loss = self.step(batch)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        test_loss = self.step(batch)
        self.log("test_loss", test_loss)

    def make_predictions(self, df, y_true=None):
        # transforms dataframe into a single batch
        if type(df) == type(pd.DataFrame()):
            x = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
        else:
            x = df
        # true y is fed but it is never used in these settings, just a model input requirement
        self.eval()
        y_prediction = self(x)
        y_prediction = y_prediction.squeeze(0)
        return y_prediction.detach().numpy()
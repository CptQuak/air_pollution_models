import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import lightning.pytorch as pl
import pandas as pd

class ARmodel(nn.Module):
    def __init__(self, seq_gen, loss_fn=F.mse_loss):
        super().__init__()
        self.seq_gen = seq_gen
        self.input_shape = int(len(self.seq_gen.input_columns_idx) * self.seq_gen.input_width, )
        self.output_shape = int(len(self.seq_gen.output_columns_idx) * self.seq_gen.output_width)
        self.linear = nn.Linear(self.input_shape, self.output_shape)
        self.loss_fn = loss_fn

    def forward(self, X):
        # reshape 3D array into 2D (batch, [a1, a2, a3, ... z1, z2, z3])
        X = X.swapaxes(1, 2).reshape(-1, self.input_shape)
        out = self.linear(X)
        return out.reshape(-1, self.seq_gen.output_width, len(self.seq_gen.output_columns_idx))

    ## LIGHTNING TRAINING UTILITY FUNCTIONS
    def configure_optimizers(self, lr=1e-4, weight_decay=1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
    
    def step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = self.loss_fn(y_hat, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('training_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_loss = self.step(batch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        test_loss = self.step(batch)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)


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

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import pandas as pd
from math import floor

class BaseLine(nn.Module):
    def __init__(self, seq_gen, ):
        super().__init__()
        self.seq_gen = seq_gen
        self.input_shape = int(len(self.seq_gen.input_columns_idx) * self.seq_gen.input_width, )
        self.output_shape = int(len(self.seq_gen.output_columns_idx) * self.seq_gen.output_width)
        self.idx_out = [i-1 for i in seq_gen.output_columns_idx]
        self.nn_layers = nn.Linear(1, 1)


    def forward(self, X):
        # (batch, inseq, infeatures)
        # (1, )
        X=X[:, :int(floor(X.shape[1]-(X.shape[1]%self.seq_gen.output_width))),self.idx_out]
        X1=X.reshape(X.shape[0], int(X.shape[1]/self.seq_gen.output_width), self.seq_gen.output_width, X.shape[2])
        X1=X1.transpose(1, 2)
        return X1.mean(2)
    
    ## LIGHTNING TRAINING UTILITY FUNCTIONS
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
            
    def configure_optimizers(self, lr=1e-4, weight_decay=1e-4):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def step(self, batch):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y_hat, y)
        return loss
    
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import lightning.pytorch as pl
import random
import pandas as pd

def init_seq2seq(module):
    if type(module) == nn.Linear: 
        nn.init.xavier_uniform_(module.weight) 
    if type(module) == nn.GRU: 
        for param in module._flat_weights_names: 
            if "weight" in param: 
                nn.init.xavier_uniform_(module._parameters[param])


class RNN_Encoder(nn.Module):
    def __init__(self, num_input, num_hidden, dropout, num_layers = 1):
        super().__init__()
        self.num_input, self.num_hidden, self.num_layers = num_input, num_hidden, num_layers

        #self.embedding = nn.Embedding(num_input, emb_dim)    
        self.dropout = dropout if self.num_layers >1 else 0
        self.rnn = nn.GRU(num_input, num_hidden, num_layers, dropout=self.dropout)
        self.apply(init_seq2seq)
    
    def forward(self, X):
        # input (length, batch, features) 
        out, hidden = self.rnn(X)
        # w przypadku LSTM mamy wyjscie z sieci w postaci  out, (hidden, cell)
        # ale w kodzie to nie zmienia
        return out, hidden


class RNN_Decoder(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, dropout, num_layers = 1):
        super().__init__()
        self.num_input, self.num_hidden, self.num_output, self.num_layers = num_input, num_hidden, num_output, num_layers

        self.dropout = dropout if self.num_layers >1 else 0
        self.rnn = nn.GRU(num_input, num_hidden, num_layers, dropout=self.dropout)
        self.linear = nn.Linear(num_hidden, num_output)
        self.apply(init_seq2seq)

    def forward(self, X, hidden_state):
        # input (length, batch, features) 
        outputs, hidden_state = self.rnn(X,  hidden_state)
        # num_hidden -> num_out
        outputs = self.linear(outputs)

        return outputs, hidden_state


class RNN_S2S(nn.Module):
    def __init__(self, seq_gen, num_hidden, enc_num_layers, dec_num_layers, dropout=.3, loss_fn = F.mse_loss):
        super().__init__()
        self.encoder = RNN_Encoder(
            num_input = len(seq_gen.input_columns_idx),
            num_hidden = num_hidden,
            num_layers = enc_num_layers,
            dropout=dropout
        )
        self.decoder = RNN_Decoder(
            num_input = len(seq_gen.output_columns_idx),
            num_hidden = num_hidden,
            num_output = len(seq_gen.output_columns_idx),
            num_layers = dec_num_layers,
            dropout = dropout
        )
        self.seq_gen = seq_gen
        self.in_seq_len, self.out_seq_len = self.seq_gen.input_width, self.seq_gen.output_width
        self.output_idx = self.seq_gen.output_columns_idx
        self.loss_fn = loss_fn
        
    def forward(self, x, y, teacher_forcing_threshold):
        # input is feed in shape (batch, length, features) -> (length, batch, features) 
        x = x.swapaxes(0, 1)
        batch_size = x.shape[1]
        # wektor wyjsc zapisywany z predykcji decodera
        outputs = torch.zeros((self.out_seq_len, batch_size, self.decoder.num_output), device = self.seq_gen.device)


        ## ENCODER
        enc_output, enc_hidden = self.encoder(x)

        ## DECODER
        # state and input preparation
        # tu jest kilka podejść, możemy wziąc ostatni, zsumować, uśrednić itd informacje z layerów encodera
        dec_hidden = torch.repeat_interleave(
            enc_hidden.sum(0, keepdim=True), self.decoder.num_layers, 0)
        # najpoplarniejsze podejscie to wziac ostatnie wejscie ostatniego timestepu encodera, ewentualnie encoder sie przesuwa jeszcze o 1 do tylu
        dec_x = torch.clone(x[-1:, :, self.output_idx])

        # generujemy kolejne timestepy wyjscia
        for t in range(0, self.out_seq_len):
            # robimy predykcje decoderem
            dec_out, dec_hidden = self.decoder(dec_x, dec_hidden)
            # zapisujemy predykcje
            outputs[t, :, :] = torch.clone(dec_out)

            # w trakcie uczenia mozemy zdecydować na zastosowanie forsowania
            # forsowanie pozwala na użycie czasem rzeczywistego wyjscia zamiast predykcji
            # co przy uczeniu poprawia nauke modelu, przy predykcji zawsze jest to wylaczone
            teacher_force = random.random() < teacher_forcing_threshold
            # regula decyzyjna 
            dec_x = (
                torch.clone(y[:, t:t+1, :].swapaxes(0, 1))
                if teacher_force else torch.clone(outputs[t:t+1, :, :])
            )

        return outputs.swapaxes(0, 1)

    ## lightning functions
    def step(self, batch):
        X, y = batch
        teacher_forcing_threshold = .5 if self.training else 0
        y_hat = self(X, y, teacher_forcing_threshold)
        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, teacher_forcing_threshold=0.5)
        self.log('training_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        val_loss = self.step(batch, teacher_forcing_threshold=0.0)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        test_loss = self.step(batch, teacher_forcing_threshold=0.0)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self, lr=0.001, weight_decay=0.0003):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
    
    def make_predictions(self, df, y_true=None):
        # transforms dataframe into a single batch
        if type(df) == type(pd.DataFrame()):
            x = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
        else:
            x = df
        # true y is fed but it is never used in these settings, just a model input requirement
        self.eval()
        y_prediction = self(x, y_true, 0.0)
        y_prediction = y_prediction.reshape(-1, len(self.output_idx))
        return y_prediction.detach().numpy()
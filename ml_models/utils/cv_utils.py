import torch
import pandas as pd
from pickle import dump
import numpy as np
import pathlib
from utils.modeling_utils import SequenceGeneratorCV, SequenceGeneratorCVPCA
import shutil
import json


def reset_weights(m):  
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss*self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def cv_fold_dataloaders(cities_dfs, seq_gen, idx_train, idx_val):
    # datalaoder with windowed data
    dataloader_train, dataloader_val = seq_gen.get_dataloaders(
        seq_gen.preprocessor.fit_transform(
            pd.concat([city.loc[idx_train, :] for city in cities_dfs], axis=0).copy()
        ), 
        seq_gen.preprocessor.transform(
            pd.concat([city.loc[idx_val, :] for city in cities_dfs], axis=0).copy()
        )
    )
    return dataloader_train, dataloader_val

def cross_validate_model(cities_dfs, model, seq_gen, cv_indices, training_path, idx_params, args):
    def save_cv_metrics(training_path, idx_params, results, loss_train, loss_val):
        metrics = pd.DataFrame(results)
        metrics['train_loss_per_epoch'], metrics['val_loss_per_epoch'] = loss_train, loss_val
        metrics.to_json(f'{training_path}/cv_metrics_{idx_params}.json', orient='records')


    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    
    results, train_losses, val_losses = [], [], []
    for i_fold, (idx_train, idx_val) in enumerate(cv_indices):
        print('-'*20)
        print(f'Fold: {i_fold+1}')
        # prepare currect fold datasets
        dataloader_train, dataloader_val = cv_fold_dataloaders(cities_dfs, seq_gen, idx_train, idx_val)
        # start training
        loss_train, loss_val = training_loop(model, dataloader_train, dataloader_val, args)
        # per fold metrics
        results.append({'fold': i_fold, 'loss_train': loss_train[-1], 'loss_val': loss_val[-1]})
        train_losses.append(loss_train.copy()), val_losses.append(loss_val.copy())
    # save per fold metrics
    save_cv_metrics(training_path, idx_params, results, train_losses, val_losses)
    
    return results, train_losses, val_losses

def find_best_performing_model(model_params, training_path):
    idx_best_model, best_perf, final_epochs = 0, np.inf, 0

    for idx_params, params in enumerate(model_params):
        # calculate mean performance on validation set
        metrics = pd.read_json(f'{training_path}/cv_metrics_{idx_params}.json')
        model_perf = np.mean(metrics['loss_val'])
        # check if better performance on val is better than current one
        if model_perf < best_perf:
            idx_best_model = idx_params
            best_perf = model_perf
            # get average number of epochs the model was trained for to avoid overfitting 
            final_epochs = int(
                np.round(
                    np.mean( [len(metrics['train_loss_per_epoch'][i]) for i in range(len(metrics))] )
                , 0)
            )

    return idx_best_model,final_epochs




def create_best_model(df, model, sg, training_path, final_epochs):
    '''
    Train modelu z o najlepszym wyniku na cv z zapisem
    '''
    
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    
    df_train = sg.preprocessor.fit_transform(df)
    dataloader_train = sg.get_dataloaders(df_train)
    # call to train model
    loss_train = train_final_model(model, dataloader_train, final_epochs)
    # save model and preprocessor
    torch.save(model.state_dict(), f'{training_path}/model.pkl')
    dump(sg.preprocessor, open(f'{training_path}/preprocesor.pkl', 'wb'))


def training_loop(model, dataloader_train, dataloader_val, args):
    num_epochs, patience, min_delta = args.NUM_EPOCHS, args.PATIENCE, args.MIN_DELTA
    model.apply(reset_weights)
    optimizer = model.configure_optimizers()
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    loss_train, loss_val = [], []
    for epoch in range(0, num_epochs):
        t_loss = single_train(model, dataloader_train, optimizer)
        loss_train.append(t_loss)

        v_loss = single_val(model, dataloader_val)
        loss_val.append(v_loss)
        if (epoch+1) % 10 == 0 or epoch==0:
            print(f'Epoch: {epoch + 1}, TRAIN LOSS: {t_loss}, VAL LOSS: {v_loss}')
        
        if early_stopper.early_stop(v_loss):
            print(f'Early stopping at epoch: {epoch+1}')
            break

    return loss_train, loss_val

def train_final_model(model, dataloader_train, num_epochs):
    print('Training best performing model:')
    model.apply(reset_weights)
    optimizer = model.configure_optimizers()

    loss_train = []
    for epoch in range(0, num_epochs):
        t_loss = single_train(model, dataloader_train, optimizer)
        loss_train.append(t_loss)

        if (epoch+1) % 10 == 0 or epoch==0:
            print(f'Epoch: {epoch + 1}, TRAIN LOSS: {t_loss}')

    return loss_train

def single_train(model, dataloader_train, optimizer):
    current_loss = 0
    model.train()
    for batch in dataloader_train:
        optimizer.zero_grad()
        loss = model.step(batch)
        current_loss += loss.item()
        try:
            loss.backward()
            optimizer.step()
        except:
            pass
    return current_loss / len(dataloader_train)
    
def single_val(model, dataloader_val):
    current_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader_val:
            loss = model.step(batch)
            current_loss += loss.item()
    return current_loss / len(dataloader_val)


def initialize_data(args):
    """
    INICJALIZACJA GENERATORÓW DANYCH
    """
    df = pd.read_csv(f'{args.DATA_PATH}/csv/six_cities.csv')
    df['dt'] = pd.to_datetime(df['dt'])
    params_seq_gen = dict(
        numeric_features = args.NUMERIC_FEATURES,
        categorical_features = args.CATEGORICAL_FEATURES,
        output_features = args.OUTPUT_FEATURES,
        input_width = args.INPUT_WIDTH,
        output_width = args.OUTPUT_WIDTH,
        normalize_features = args.NORMALIZE_FEATURES,
        device = args.DEVICE,
    )
    seq_gen = SequenceGeneratorCV(**params_seq_gen)
    seq_gen.init_preprocessor(df)
    seq_gen_pca = SequenceGeneratorCVPCA(**params_seq_gen)
    seq_gen_pca.init_preprocessor(df)
    return df, seq_gen, seq_gen_pca

def init_training(args, df, model_name, sg, model_params):
    """
    dataframe per city, indexy i okreslenie model pathu
    """
    cities_dfs, cv_indices = sg.split_data(df, args.CV_FOLDS, args.CV_INCREMENT)
    model_path = f'{args.STORING_DIR}/{model_name}'
    training_path = f'{model_path}/training_{args.TRAINING_TYPE}'
    pathlib.Path(training_path).mkdir(parents=True, exist_ok=True)

    with open(f'{training_path}/params.txt', 'w') as fp:
        for param in model_params:
            fp.write(f'{param}\n')
    return cities_dfs, cv_indices, model_path, training_path

def save_best_model_metrics(model_params, training_path, idx_best_model):
    best_model_metrics = pd.read_json(f'{training_path}/cv_metrics_{idx_best_model}.json')
    best_model_metrics = {
            'loss_val' : best_model_metrics['loss_val'].tolist(),
            'params': model_params[idx_best_model],
        }
    json.dump(best_model_metrics, open(f'{training_path}/best_model_info.json', 'w'))
    return best_model_metrics

def move_model_files(model_path, training_path, idx_best_model):
    shutil.copyfile(f'{training_path}/model.pkl', f'{model_path}/model.pkl')
    shutil.copyfile(f'{training_path}/preprocesor.pkl', f'{model_path}/preprocesor.pkl')
    shutil.copyfile(f'{training_path}/cv_metrics_{idx_best_model}.json', f'{model_path}/cv_metrics.json')
    shutil.copyfile(f'{training_path}/best_model_info.json', f'{model_path}/best_model_info.json')

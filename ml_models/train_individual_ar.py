from utils import cv_utils
import models
import pandas as pd
import numpy as np
import torch
import os
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"


class Args:
    def __init__(self):
        self.STORING_DIR = "trained_models_individual"
        self.DATA_PATH = "../data"
        self.INPUT_WIDTH, self.OUTPUT_WIDTH = int(7*24), 48
        self.NORMALIZE_FEATURES = ['humidity', 'clouds.all', 'rain.1h', 'snow.1h', 'co', 'no', 'no2', 'so2', 'pm2_5', 'pm10', 'nh3']
        self.NUMERIC_FEATURES = ['day_sin',	'day_cos', 'week_sin', 'week_cos', 'month_sin','month_cos', 'temp', 'pressure', 'humidity', 'wind.x', 'wind.y', 'clouds.all', 'rain.1h', 'snow.1h', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        self.CATEGORICAL_FEATURES = ['state']
        self.OUTPUT_FEATURES = ['co', 'no', 'no2', 'o3', 'so2',	'pm2_5', 'pm10', 'nh3']
        self.DEVICE = DEVICE 
        # CV OPTIONS
        self.CV_FOLDS, self.CV_INCREMENT = 5, False
        self.NUM_EPOCHS, self.PATIENCE, self.MIN_DELTA = 250, 15, 1.01
        self.TRAINING_TYPE = 'incremental' if self.CV_INCREMENT else 'moving'


def main(args):
    #### INICJALIZACJA GENERATORÓW DANYCH
    OUTPUT_FEATURES = ['co', 'no', 'no2', 'o3', 'so2',	'pm2_5', 'pm10', 'nh3']

    # model dla kazdej zmiennej oddzielnie
    for i in OUTPUT_FEATURES:
        args.OUTPUT_FEATURES = [i]
        df, seq_gen, seq_gen_pca = cv_utils.initialize_data(args)
    
        ar_params = [{}]
        list_of_models = [
            (f'ar_seq2seq_{i}', models.ARmodel, seq_gen, ar_params),
        ]

        # TRAINING EACH MODEL
        for model_name, _model, sg, model_params in list_of_models:
            # INITIALIZE SEQ_GEN AND PATHS
            cities_dfs, cv_indices, model_path, training_path = cv_utils.init_training(args, df, model_name, sg, model_params)
            # CROSS VALIDATE MODEL ON ON EACH POSSIBLE SET OF PARAMETERS
            for idx_params, params in enumerate(model_params):
                print(f'Currently training: {model_name} with {params}')
                model = _model(sg, **params)
                cv_utils.cross_validate_model(cities_dfs, model, sg, cv_indices, training_path, idx_params, args)

            # FIND BEST PERFORMING MODEL ON CV
            idx_best_model, final_epochs = cv_utils.find_best_performing_model(model_params, training_path)
            # TRAIN AND SAVE BEST MODEL
            print('-'*20)
            print(f'Best {model_name} params: {model_params[idx_best_model]}')
            model = _model(sg, **model_params[idx_best_model])
            cv_utils.create_best_model(df, model, sg, training_path, final_epochs)
            best_model_metrics = cv_utils.save_best_model_metrics(model_params, training_path, idx_best_model)


            #### SPRAWDZENIE CZY AKTUALNY JEST LEPSZY
            # compare_and_save_model(model_path, training_path, idx_best_model, best_model_metrics)
            if 'model.pkl' not in os.listdir(model_path):
                print('Existing model not found, copying current best')
                cv_utils.move_model_files(model_path, training_path, idx_best_model)
            else:
                with open(f'{model_path}/best_model_info.json', 'r') as f: old_model_info = json.load(f)
                print(f"Old model: {old_model_info['loss_val']}\nNew model: {best_model_metrics['loss_val']}")
                if (
                    np.mean(old_model_info['loss_val']) > np.mean(best_model_metrics['loss_val'])
                ):
                    print('Inserting new model')
                    cv_utils.move_model_files(model_path, training_path, idx_best_model)
                else:
                    print('Keeping older model')


if __name__ == '__main__':
    args = Args()
    for i in [False, True]:
        args.CV_INCREMENT = i
        args.TRAINING_TYPE = 'incremental' if args.CV_INCREMENT else 'moving'
        print('\n'*3)
        print(f'CROSS VALIDATION MODE: {args.TRAINING_TYPE}')
        main(args)
    print('Done training models')
# STUPID DESIGN
import sys
from utils.predictions import load_latest_data, model_prediction
from utils.modeling_utils import SequenceGeneratorCV, SequenceGeneratorCVPCA
from models import ARmodel, RNN_S2S, DNN_model, BaseLine
from pickle import load
import torch
import json
DEVICE = 'cpu'


class Args:
    def __init__(self):
        self.MODELS_PATH = "trained_models"
        self.DATA_PATH = "../data"
        self.INPUT_WIDTH, self.OUTPUT_WIDTH = int(7*24), 48
        self.NORMALIZE_FEATURES = ['humidity', 'clouds.all', 'rain.1h', 'snow.1h', 'co', 'no', 'no2', 'so2', 'pm2_5', 'pm10', 'nh3']
        self.NUMERIC_FEATURES = ['day_sin',	'day_cos', 'week_sin', 'week_cos', 'month_sin','month_cos', 'temp', 'pressure', 'humidity', 'wind.x', 'wind.y', 'clouds.all', 'rain.1h', 'snow.1h', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        self.CATEGORICAL_FEATURES = ['state']
        self.OUTPUT_FEATURES = ['co', 'no', 'no2', 'o3', 'so2',	'pm2_5', 'pm10', 'nh3']
        self.DEVICE = DEVICE 
        # CV OPTIONS
        self.CV_FOLDS, self.CV_INCREMENT = 5, False
        self.NUM_EPOCHS, self.PATIENCE, self.MIN_DELTA = 250, 25, 1.01
        self.TRAINING_TYPE = 'incremental' if self.CV_INCREMENT else 'moving'

def main(args):
    # preparation for model initalization
    df = load_latest_data(args.DATA_PATH+'/csv/six_cities.csv', args.INPUT_WIDTH)
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

    models = [
        ('ar_seq2seq', ARmodel, seq_gen),
        ('rnn_seq2seq', RNN_S2S, seq_gen),
        ('dnn_seq2seq', DNN_model, seq_gen),
        ('ar_seq2seq_pca', ARmodel, seq_gen_pca),
        ('baseline', BaseLine, seq_gen),
    ]

    for m_name, model, sg in models:
        print(f'Generating predictions from {m_name}')
        with open(f'{args.MODELS_PATH}/{m_name}/best_model_info.json', 'r') as f: model_metrics = json.load(f)
        preprocessor = load(open(f'{args.MODELS_PATH}/{m_name}/preprocesor.pkl', 'rb'))
        model = model(sg, **model_metrics['params'])
        model.load_state_dict(torch.load(f'{args.MODELS_PATH}/{m_name}/model.pkl'))
        model.eval()
        model_prediction(df, m_name, model, preprocessor, sg)

    print('All predictions generated')

if __name__ == '__main__':
    args = Args()
    main(args)
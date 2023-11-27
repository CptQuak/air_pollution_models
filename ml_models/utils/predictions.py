import pandas as pd
import numpy as np
import torch

def load_latest_data(path, input_width):
    """
    Loads latest data for each city for given input width
    """
    df = pd.read_csv(path)
    if 'dt' not in df.columns and 'city' not in df.columns:
        raise Exception('Wrong file')
    
    df['dt'] = pd.to_datetime(df['dt'])

    cities = df['city'].unique()

    dfs = []
    for city in cities:
        df_temp = (
            df[df['city'] == city]
            .reset_index(drop=True)
            .iloc[-input_width:, :]
        )
        dfs.append(df_temp)

    df = pd.concat(dfs, axis=0)
    return df


def model_prediction(df, m_name, model, preprocessor, sg):
    """
    1. Loads model and preprocessor
    2. Makes prediction for each city
    3. Saves prediction to csv
    """
    df_predictions = []

    for city in df.city.unique():
        df_pred_temp = city_prediction(df, model, sg,  preprocessor, city)            
        df_predictions.append(df_pred_temp)

    df_predictions = pd.concat(df_predictions, axis=0)
    df_predictions.to_csv(
            f'../data/predictions/{m_name}.csv', index=False
        )


def city_prediction(df, model, seq_gen, preprocessor, city):
    """
    Transforms data to make predictions, makes prediction, create dataframe with time labels and city name
    """
    df_temp = df[df.city == city]
    latest_date = df_temp['dt'].max()
    df_temp = preprocessor.transform(df_temp).iloc[:, seq_gen.input_columns_idx]
    y_prediction = model.make_predictions(df_temp)
    y_prediction_rescaled = rescale_predictions(
                y_prediction, 
                preprocessor, 
                seq_gen.output_width, 
                seq_gen.output_features
            )
    df_pred_temp = pd.DataFrame(y_prediction_rescaled, columns=seq_gen.output_features)
    
    dt = pd.date_range(latest_date + pd.Timedelta(hours=1), periods=seq_gen.output_width, freq='H')
    df_pred_temp['dt'] = dt
    df_pred_temp.insert(0, 'dt', df_pred_temp.pop('dt'))

    df_pred_temp['city'] = city
    df_pred_temp.insert(0, 'city', df_pred_temp.pop('city'))
    return df_pred_temp


def rescale_predictions(y_prediction, preprocessor, output_width, output_columns):
    """
    1. prepares np.array (output_width, output_columns)
    2. Calls to find related transforer with each column
    3. Rescales data to original scale and saves in array column
    """ 
    y_prediction_rescaled = np.zeros((output_width, len(output_columns)))
    for idx, col_name in enumerate(output_columns):
        transformer, transformer_features = find_transformer(col_name, preprocessor)
        y_prediction_rescaled[:, idx] = rescale_feature(y_prediction, idx, col_name, transformer, transformer_features)

    return y_prediction_rescaled


def find_transformer(col_name, preprocessor):
    # 1 - 3 to zakres transformerow danych
    """
    1. Checks idx 1 - 3 of transformers to find related preprocessor
    2. return stransformer and related features when found
    """
    for j in range(1, 4):
        transformer = preprocessor[0].transformers_[j][1]

        if col_name in transformer.feature_names_in_:
            transformer_features = transformer.feature_names_in_
            return transformer, transformer_features
        
    raise Exception('This feature doesnt exist in any transformer')
        
        
def rescale_feature(y_prediction, idx, col_name, transformer, transformer_features):
    """
    1. Creates temporal dataframe to rescale individual column
    2. Returns column coresponding to particular feature
    """
    temp_df = pd.DataFrame(np.column_stack(
        [y_prediction[:, idx] for k in transformer_features]
    ), columns=transformer_features)
    # chwilowy dataframe bo taki wymog transformera, 
    # z tego bierzemy tylko kolumne targetu
    temp_df = pd.DataFrame(
        transformer.inverse_transform(temp_df),
        columns=transformer_features
    )
    return temp_df[col_name].to_numpy()

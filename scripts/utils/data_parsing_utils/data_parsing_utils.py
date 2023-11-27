import pandas as pd
import json
import re

def read_city_data(city, path=''):
    '''
    Reads weather pollution data for a city
    '''
    with open(f'{path}/{city}/weather_data.json','r') as f:
        w_data = json.loads(f.read())
    with open(f'{path}/{city}/pollution_data.json','r') as f:
        p_data = json.loads(f.read())

    return w_data, p_data

def read_city_names(path):
    '''
    Read city names
    '''
    with open(f'{path}/city_list.txt', 'r', encoding='utf-8') as f:
        city_list = f.read().split('\n')
    return city_list


def weather_json_df(w_data):
    '''
    
    '''
    # load from json, add city column, convert timestamp to pandas date_time
    df_weather = pd.json_normalize(w_data)
    df_weather['dt'] = pd.to_datetime(df_weather.dt, unit='s')
    # remove main. from feature names
    pattern_main = re.compile(r'^main\.')
    df_weather.columns = [pattern_main.sub('', s) for s in df_weather.columns]
    # optinal parameters because in large cities its an average from few sensors
    df_weather = df_weather.drop(columns=['temp_min', 'temp_max', 'weather', 'wind.gust'])
    return df_weather

def pollution_json_df(p_data):
    # load from json, add city column, convert timestamp to pandas date_time
    df_polution = pd.json_normalize(p_data)
    df_polution['dt'] = pd.to_datetime(df_polution.dt, unit='s')

    # remove the components. and  main. from column names,
    pattern_components = re.compile(r'^components\.')
    pattern_main = re.compile(r'^main\.')

    new_pollution_columns = [pattern_components.sub('', s) for s in df_polution.columns]
    new_pollution_columns = [pattern_main.sub('', s) for s in new_pollution_columns]
    df_polution.columns = new_pollution_columns
    df_polution = df_polution.drop(columns=['aqi'])
    return df_polution

def merge_weather_pollution_df(df_weather, df_pollution, city):
    df_merged = pd.merge(df_weather, df_pollution, 'left', left_on='dt', right_on='dt')
    df_merged.insert(0, column='city', value = city)
    df_merged = df_merged.sort_values(by='dt')
    return df_merged
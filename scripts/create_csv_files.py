import pandas as pd
import numpy as np
import json
import re
import utils.data_parsing_utils as dp_utils
data_path = '../data'

with open(f'{data_path}/cities_lat_lon_data.json', 'r', encoding='utf-8') as f:
    data_cities = json.loads(f.read())

def add_state(city):
    state = data_cities[city]['state']
    return state

def data_preparation(df):
     # change metrics of temperature (from Kelvin to Celsius)
    df['temp'] = df['temp'].apply(lambda t: t-273.15)
    df['feels_like'] = df['feels_like'].apply(lambda t: t-273.15)

    # change wind.speed and wind.deg to wind velocity vector which has components wind.x and wind.y
    velocity = df.pop('wind.speed')
    # convert degress to radians
    direction_rad = df.pop('wind.deg')*np.pi / 180
    # calculate the wind x and y components
    df['wind.x'] = velocity*np.cos(direction_rad)
    df['wind.y'] = velocity*np.sin(direction_rad)

    # time transformation to sin and cos signals
    timestamp_s = df['dt'].map(pd.Timestamp.timestamp)
    #day cycle
    day = 24*60*60
    df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))

    #week cycle
    week = day*7
    df['week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))

    #month cycle
    month = day*(365.25/12)
    df['month_sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df['month_cos'] = np.cos(timestamp_s * (2 * np.pi / month))

    # change columns order
    list_cols = ['city','dt','day_sin','day_cos','week_sin','week_cos','month_sin','month_cos','temp','feels_like','pressure','humidity',\
                'wind.x','wind.y','clouds.all','rain.1h','snow.1h','co','no','no2','o3','so2','pm2_5','pm10','nh3']
    df = df[list_cols]

    # add state
    df['state'] = df['city'].apply(add_state)

    return df

def merge_jsons():
    df_cities = []
    city_list = dp_utils.read_city_names(data_path)

    for city in city_list:
        # load jsons
        w_data, p_data = dp_utils.read_city_data(city, data_path)
        # create weather dataframe
        df_weather = dp_utils.weather_json_df(w_data)
        # create airplooution dataframe
        df_pollution = dp_utils.pollution_json_df(p_data)
        # merge weather and airpollution
        df_merged = dp_utils.merge_weather_pollution_df(df_weather, df_pollution, city)

        # change cols type to numeric
        for col in df_merged:
            if col not in ['city','dt']:
                df_merged[col] = pd.to_numeric(df_merged[col])

        # fill missing values in rain and snow cols with 0
        values = {"rain.1h": 0, "snow.1h": 0}
        df_merged = df_merged.fillna(value=values)

        # fill missing values in weather data
        df_tmp = df_merged.loc[:, df_merged.columns[2:]]
        df_tmp = df_tmp.interpolate(method = 'linear', limit_direction = 'both', limit = 40) # max 4 (?) missing values in a row

        df_merged.update(df_tmp)

        df_cities.append(df_merged.copy())
    # combine all city dataframes
    df_cities = pd.concat(df_cities)
    
    df_cities = data_preparation(df_cities)

    df_cities.to_csv(f'{data_path}/csv/six_cities.csv', index=False)
    print('Saved!')


if __name__ == "__main__":
    merge_jsons()
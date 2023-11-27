import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import utils.api_utils as api_utils
import asyncio
import calendar
import json
# env variables
import os 
from dotenv import load_dotenv

# loading api key
load_dotenv()
KEY = os.getenv('owm_api_key')
data_path = '../data'
SECONDS_IN_HOUR = 3600


def check_api_key():
    # check if API KEY exists in .env
    if KEY == None:
        raise Exception("API KEY is missing")

def is_up_to_date(start_date, end_date):
    # checks if stored data is up to date
    timestamp_end = calendar.timegm(end_date.utctimetuple()) 
    timestamp_start = calendar.timegm(start_date.utctimetuple())
    print(timestamp_end, timestamp_start)
    # it checks if the new start date can fetch any new data
    return  timestamp_end - timestamp_start < 0 

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
        # last collected date in unix time
        last_date_timestamp_p = data[-1]['dt']
        # start date to download new data (1 hour after last date) in unix time
        s_timestamp_p = last_date_timestamp_p + SECONDS_IN_HOUR
        start_time = datetime.fromtimestamp(s_timestamp_p, timezone.utc)
    return data, start_time
    
def save_to_json(path, new_data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f) 


async def fetch_data():
    ## call for cities geo-coding
    city_lat_lon = await api_utils.get_geo_coding(data_path, KEY)
    # loading bar variables
    iteration_count, total_count = 0, len(city_lat_lon.keys()) 

    # converting to unix time for OWM API
    end_date = datetime.now(timezone.utc)
    
    for city, value in city_lat_lon.items():
        print('------------------')
        iteration_count += 1
        print(f'{iteration_count}/{total_count}\t {city}')
        lat, lon = value['lat'], value['lon']

        ## check if city data already exists, create folder if not
        city_path = Path(f'{data_path}/{city}')
        dataFolderExists = city_path.exists()

        if dataFolderExists:
            pollution_data, start_date = read_json(f'{city_path}/pollution_data.json')
            weather_data, start_date = read_json(f'{city_path}/weather_data.json')
            
            if is_up_to_date(start_date, end_date):
                print(f'{city} data is up to date!')
                continue
        else:
            city_path.mkdir(parents=True)
            start_date = end_date - timedelta(days=364)

        
        

        ## get airpollution data and save
        print('> air pollution data')
        new_pollution_data = await api_utils.call_pollution_data(KEY, lat, lon, start_date, end_date)

        if dataFolderExists:
            pollution_data.extend(new_pollution_data) # extends existing json
        else:
            pollution_data = new_pollution_data
        save_to_json(f'{data_path}/{city}/pollution_data.json', pollution_data)
        
                
        ## get weather data and save
        print('> weather data')
        new_weather_data = await api_utils.call_weather_data(KEY, lat, lon, start_date, end_date)

        if dataFolderExists:
            weather_data.extend(new_weather_data) # extends existing json
        else:
            weather_data = new_weather_data
        save_to_json(f'{data_path}/{city}/weather_data.json', weather_data)


if __name__ == "__main__":
    check_api_key()
    asyncio.run(fetch_data())
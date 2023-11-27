import requests
import json
import asyncio
import calendar
from datetime import datetime, timedelta, timezone


async def get_pollution_data(KEY, lat, lon, start, end):
    url = f'http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={KEY}'
    response = requests.get(url)
    return response.json()


async def get_weather_data(KEY, lat, lon, start, end):
    url = f'https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={KEY}'
    response = requests.get(url)
    return response.json()

async def get_geo(KEY, city):
    url = f'http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={KEY}'
    response = requests.get(url)
    return response.json()


async def get_geo_coding(data_path, KEY):
    '''
    Creates a task to call geo_coding OWM API and writes to json file
    return dictionary of geo data for a particular city
    '''
    # input is a txt file with city names
    with open(f'{data_path}/city_list.txt', 'r', encoding='utf-8') as f:
        city_list = f.read().split('\n')

    city_lat_lon = {city: "" for city in city_list}

    for city in city_list:
        # get lat lan for given city
        geo_data = asyncio.create_task(get_geo(KEY, city))
        geo_data = await geo_data
        # data are returned as a list so we take first element to have a clean object
        try:
            geo_data = geo_data[0]
        except:
            raise Exception(geo_data)
        
        # update city data
        city_lat_lon[city] = {
            "lat": geo_data['lat'], 
            "lon": geo_data['lon'], 
            "state": geo_data['state']
        }

    # na wszelki wypadek zapis do pliku
    with open(f'{data_path}/cities_lat_lon_data.json', 'w', encoding='utf-8') as f:
        json.dump(city_lat_lon, f)
    return city_lat_lon



async def call_pollution_data(KEY, lat, lon, start, end):
    '''
    Creates a task to call air pollution data OWM API and writes to json file
    '''
    s_timestamp = calendar.timegm(start.utctimetuple()) 
    e_timestamp = calendar.timegm(end.utctimetuple()) 
    pol_data = asyncio.create_task(
        get_pollution_data(KEY, lat, lon, s_timestamp, e_timestamp)
    )
    pol_data = await pol_data
    return pol_data['list']


async def call_weather_data(KEY, lat, lon, start, end):
    '''
    Creates a task to call waether data OWM API and writes to json file
    '''
    # an array to store sequences of data from historical weather api
    we_dat_full = []
    
    stop_flag = False
    temp_start = start
    temp_end = temp_start + timedelta(days=7)
    # weather data
    while(True):
        # time to unix timestamps
        s_timestamp = calendar.timegm(temp_start.utctimetuple()) 
        e_timestamp = calendar.timegm(temp_end.utctimetuple()) 

        if temp_end > end:
            # checking out if we arent looking too far forward in time, 
            # in this case we change the end time to given END and set the stop flag
            e_timestamp = calendar.timegm(end.utctimetuple())
            if e_timestamp == s_timestamp:
                break
            stop_flag = True

        # create a task to call waether data api
        weather_data = asyncio.create_task(
            get_weather_data(KEY, lat, lon, s_timestamp, e_timestamp)
        )
        weather_data = await weather_data

        # in a case of faulty response retry same call
        if 'list' not in weather_data.keys():
            print(weather_data)
            continue
        # concatenating weekly lists into single full list
        we_dat_full.extend(weather_data['list'])

        if stop_flag:
            break

        # moving through every week upwards
        temp_start = temp_end
        temp_end = temp_start + timedelta(days=7)

    return we_dat_full

async def call_weather_data_old(KEY, lat, lon, start, end):
    '''
    Creates a task to call waether data OWM API and writes to json file
    '''
    # an array to store sequences of data from historical weather api
    we_dat_full = []
    
    stop_flag = False
    temp_start = start
    temp_end = temp_start + timedelta(days=7)
    # weather data
    while(True):
        # time to unix timestamps
        s_timestamp = calendar.timegm(temp_start.utctimetuple()) 
        e_timestamp = calendar.timegm(temp_end.utctimetuple()) 

        if temp_end > end:
            # checking out if we arent looking too far forward in time, 
            # in this case we change the end time to given END and set the stop flag
            s_timestamp = calendar.timegm((end - timedelta(days=6)).utctimetuple()) 
            e_timestamp = calendar.timegm(end.utctimetuple())
            stop_flag = True

        # create a task to call waether data api
        weather_data = asyncio.create_task(
            get_weather_data(KEY, lat, lon, s_timestamp, e_timestamp)
        )
        weather_data = await weather_data

        # in a case of faulty response retry same call
        if 'list' not in weather_data.keys():
            print(weather_data)
            continue
        # concatenating weekly lists into single full list
        we_dat_full.extend(weather_data['list'])

        if stop_flag:
            break

        # moving through every week upwards
        temp_start = temp_end
        temp_end = temp_start + timedelta(days=7)

    return we_dat_full
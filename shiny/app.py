from shiny import App, render, ui, Inputs, Outputs, Session, reactive
import pandas as pd
import time
import matplotlib.pyplot as plt
from htmltools import css
from shinywidgets import output_widget, reactive_read, register_widget
import ipyleaflet as L
import ipywidgets as widgets
import json
from datetime import date, datetime, timedelta
from io import StringIO
from shiny_download import get_url
import asyncio

# dane
url_ar = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/predictions/ar_seq2seq.csv'
url_ar_pca = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/predictions/ar_seq2seq_pca.csv'
url_dnn = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/predictions/dnn_seq2seq.csv'
url_rnn = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/predictions/rnn_seq2seq.csv'
url_baseline = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/predictions/baseline.csv'
url_six_cities = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/csv/six_cities.csv'
url_lat_lon = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/cities_lat_lon_data.json'
url_city_list = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/city_list.txt'
url_var_names = r'https://raw.githubusercontent.com/uni-lec-laboratory/lab-2223-group1-website/main/data/variable_names.txt'

app_ui = ui.page_fluid(
    ui.navset_tab_card(
    
        ui.nav("Maps", 
               
            ui.panel_title(ui.tags.h1("Awesome map")),

            ui.layout_sidebar(
            ui.panel_main(
                ui.panel_well(
                    # mapa
                    output_widget("map",height='450px'),
                )),
            ui.panel_sidebar(
                # wybór daty
                ui.input_date('date',
                              'Choose date',
                              value = "2023-05-18"
                              ),
                # wybór zmiennej
                ui.input_radio_buttons('var',
                                       "Choose variables",
                                        choices = [''],
                                    ),
                # napis z informacją
                ui.output_ui('map_bounds')
            )
            ),
        ),
        ui.nav("Pollution", 
               
            ui.panel_title(ui.tags.h1("Pollution forecast")),

            ui.layout_sidebar(

                ui.panel_sidebar(
                        # wybór miasta
                        ui.panel_well(
                            ui.input_select('slct_city',
                                            "Choose city",
                                            choices = [''])
                            ),
                        # wybór zmiennych
                        ui.panel_well(
                            ui.input_checkbox_group('slct_var',
                                                    "Choose variables",
                                                    choices=[''])
                        ),
                        # wybór dat
                        ui.panel_well(
                            ui.input_date_range('dates',
                                            "Select date:",
                                            weekstart = 1,
                                            startview = "year"
                                            ),
                        ),
                        width=3
                ),

            ui.panel_main(
                    ui.panel_well(
                        # wybór modelu
                        ui.input_radio_buttons("model",
                                               "Choose model",
                                               choices = ['']
                                               ),
                        # wykresy
                        ui.output_plot("plots",height='450px'),
                    ),
                    width=9
                ),
            ),
        ),
    ),
)

def server(input: Inputs, output: Outputs, session: Session):

    reactive_df_predicted_ar = reactive.Value(pd.DataFrame())
    reactive_df_predicted_ar_pca = reactive.Value(pd.DataFrame())
    reactive_df_predicted_dnn = reactive.Value(pd.DataFrame())
    reactive_df_predicted_rnn = reactive.Value(pd.DataFrame())
    reactive_df_predicted_baseline = reactive.Value(pd.DataFrame())

    reactive_df_data = reactive.Value(pd.DataFrame())
    reactive_df_city = reactive.Value(pd.DataFrame())
    reactive_df_var = reactive.Value(pd.DataFrame())
    reactive_city_lat_lon = reactive.Value()
    
    @reactive.Effect
    async def _():
        # lista miast
        response = await get_url(url_city_list, 'string')
        data = StringIO(response.data)
        reactive_df_city.set(pd.read_csv(data, header=0))
        city_options = [item[1]['city'] for item in (reactive_df_city().to_dict('index')).items()]
        ui.update_select(
            'slct_city',
            choices = city_options
        )
        
        # lista zmiennych
        response = await get_url(url_var_names, 'string')
        data = StringIO(response.data)
        reactive_df_var.set(pd.read_csv(data, header=0))
        df_var = reactive_df_var()[7:]
        df_var.columns = ['var_name','var_value']
        df_var_items = (df_var.to_dict('index')).items()
        var_options = {item[1]['var_value'] : item[1]['var_name'] for item in df_var_items}
        var_options2 = {item[1]['var_value'] : item[1]['var_name'] for item in df_var_items if item[1]['var_value'] not in ['nh3','no']}

        ui.update_radio_buttons(
            'var',
            choices = var_options2
        )

        ui.update_checkbox_group(
            'slct_var',
            choices = var_options
        )

        # lista dostępnych modeli
        models = {'ar_pca':'AR PCA','ar':'AR','dnn':'DNN','rnn':'RNN','baseline':'Baseline'}
        ui.update_radio_buttons(
            'model',
            choices = models,
            inline=True
        )
        # dane
        response = await get_url(url_six_cities, 'string')
        data = StringIO(response.data)
        reactive_df_data.set(pd.read_csv(data, header=0))
        reactive_df_data()['dt'] = pd.to_datetime(reactive_df_data()['dt'])

        response = await get_url(url_ar, 'string')
        data = StringIO(response.data)
        reactive_df_predicted_ar.set(pd.read_csv(data, header=0))
        reactive_df_predicted_ar()['dt'] = pd.to_datetime(reactive_df_predicted_ar()['dt'])

        # daty
        # daty początkowa i końcowa w danych
        start = min(reactive_df_data()['dt']).date()
        end1 = max(reactive_df_data()['dt']).date()
        end = max(reactive_df_predicted_ar()['dt']).date()

        ui.update_date(
            'date',
            value=end1 - timedelta(days=1),
            min= start,
            max= end1 - timedelta(days=1)
        )

        ui.update_date_range(
            'dates',
            start=start,
            end=end,
            min=start,
            max=end,
        )

        # współrzędne geograficzne miast
        response = await get_url(url_lat_lon, 'json')
        data = response.data
        reactive_city_lat_lon.set(data)

        # wczytanie predykcji
        response = await get_url(url_ar_pca, 'string')
        data = StringIO(response.data)
        reactive_df_predicted_ar_pca.set(pd.read_csv(data, header=0))
        reactive_df_predicted_ar_pca()['dt'] = pd.to_datetime(reactive_df_predicted_ar_pca()['dt'])

        response = await get_url(url_rnn, 'string')
        data = StringIO(response.data)
        reactive_df_predicted_rnn.set(pd.read_csv(data, header=0))
        reactive_df_predicted_rnn()['dt'] = pd.to_datetime(reactive_df_predicted_rnn()['dt'])

        response = await get_url(url_dnn, 'string')
        data = StringIO(response.data)
        reactive_df_predicted_dnn.set(pd.read_csv(data, header=0))
        reactive_df_predicted_dnn()['dt'] = pd.to_datetime(reactive_df_predicted_dnn()['dt'])

        response = await get_url(url_baseline, 'string')
        data = StringIO(response.data)
        reactive_df_predicted_baseline.set(pd.read_csv(data, header=0))
        reactive_df_predicted_baseline()['dt'] = pd.to_datetime(reactive_df_predicted_baseline()['dt'])

    # funkcja tworząca wycinek danych w zależności od wybranego miasta i czasu
    @reactive.Calc
    def df_section():
        if(input.model() == 'ar_pca'):  # dane predykcji w zależności od wyboru użytkownika
            df_predicted = reactive_df_predicted_ar_pca()      
        elif(input.model() == 'ar'):
            df_predicted = reactive_df_predicted_ar()    
        elif(input.model() == 'dnn'):
            df_predicted = reactive_df_predicted_dnn() 
        elif(input.model() == 'rnn'):
            df_predicted = reactive_df_predicted_rnn()
        elif(input.model() == 'baseline'):
            df_predicted = reactive_df_predicted_baseline()   

        df_predicted['predicted'] = 1   # dodatkowa kolumna wskazująca czy jest to predykcja czy nie
        df_data = reactive_df_data()
        df_data['predicted'] = 0
        df = pd.concat([df_data,df_predicted],ignore_index=True)

        df = df[df["city"] == input.slct_city()] # wycinek z wybranym miastem

        df['dt_unix'] = df['dt'].apply(lambda row: time.mktime(row.timetuple())) # dodanie nowej kolumny z czasem uniksowym

        start =  input.dates()[0] # wybrane daty
        end = input.dates()[1]
        start_date = time.mktime(start.timetuple()) # zamiana wybranych dat na czas uniksowy
        end_date = time.mktime(end.timetuple())

        mask = (df['dt_unix'] >= start_date) & (df['dt_unix'] <= end_date) # wycinek z wybranym czasem
        df = df.loc[mask]

        return df
    
    # funkcja tworząca wykresy dla wybranych zmiennych
    @output
    @render.plot
    def plots():
        df =  df_section()
        df_var = reactive_df_var()[7:]
        df_var.columns = ['var_name','var_value']
        df_var_items = (df_var.to_dict('index')).items()
        var_options = {item[1]['var_value'] : item[1]['var_name'] for item in df_var_items}

        plt.style.use('ggplot')
        fig, ax = plt.subplots(1, 2, figsize=(20, 8))
        ax = ax.ravel()
        for var, i in zip(input.slct_var(), range(0,2)):
            
            # wydzielenie zbirow historycznych i predykcji
            mask1 = df['predicted'] == 0 
            mask2 = df['predicted'] == 1

            ax[i].plot(df['dt'][mask1],df[var][mask1], color='#569DAA')
            ax[i].plot(df['dt'][mask2],df[var][mask2], color='#F2A950')
            ax[i].fill_between(df['dt'][mask1],df[var][mask1], color='#569DAA', alpha=0.5) # wypelnienie obszaru pod wykresem
            ax[i].fill_between(df['dt'][mask2],df[var][mask2], color='#F2A950', alpha=0.5)
            
            # ustawienie parametrów wykresu
            ax[i].set_title(var_options[var])
            ax[i].set_xlabel('date')
            ax[i].set_xlim(input.dates()[0], input.dates()[1])
            ax[i].set_ylim(bottom=0)
            fig.autofmt_xdate()

        return fig

    # MAPA
    # Initialize and display when the session starts (1)
    map = L.Map(center=(52.565162, 19.252522), zoom=6, scroll_wheel_zoom=True, close_popup_on_click=False)
    # Add a distance scale
    map.add_control(L.leaflet.ScaleControl(position="bottomleft"))
    register_widget("map", map)

    # funkcja tworząca wycinek danych w zależności od wybranej daty
    @reactive.Calc
    def df_time_section():
        df_copy = reactive_df_data()

        df_copy['dt_unix'] = df_copy['dt'].apply(lambda row: time.mktime(row.timetuple())) # dodanie nowej kolumny z czasem uniksowym

        d =  input.date()
        data = time.mktime(d.timetuple()) # zamiana wybranych dat na czas uniksowy
        mask = (df_copy['dt_unix'] >= data) # wycinek z wybranym czasem
        df_copy = df_copy.loc[mask]

        return df_copy
    
    # funkcja do znaczników na mapie
    @reactive.Effect
    def _():
        df_time_sec = df_time_section()
        data_cities_lat_lon = reactive_city_lat_lon()

        # Create markers
        for city in data_cities_lat_lon:
            df = df_time_sec[df_time_sec["city"] == city]
            
            start_t = pd.to_datetime(df['dt'].values[0]) + timedelta(hours = 8)     # średni poziom zanieczyszczenia 8.00 - 20.00
            end_t = pd.to_datetime(df['dt'].values[0]) + timedelta(hours = 20)

            df.set_index('dt', inplace=True)
            df_new = df[start_t:end_t]

            icons_urls = {'good':'https://cdn-icons-png.flaticon.com/512/725/725070.png',       # ikony w zależności od poziomu zanieczysczenia
                          'fair':'https://cdn-icons-png.flaticon.com/512/725/725105.png',
                          'moderate':'https://cdn-icons-png.flaticon.com/512/725/725085.png',
                          'poor':'https://cdn-icons-png.flaticon.com/512/725/725099.png',
                          'very_poor':'https://cdn-icons-png.flaticon.com/512/725/725117.png'}
            
            var_limits = {'co':[4400,9400,12400,15400],         # pozimy zanieczyszczenia poszczególnymi związkami
                          'no2':[40,70,150,200],
                          'o3':[60,100,140,180],
                          'so2':[20,80,250,350],
                          'pm2_5':[10,25,50,75],
                          'pm10':[20,50,100,200]}

            if (input.var() != ''):
                avg_var = df_new[input.var()].mean()
                if avg_var < var_limits[input.var()][0]:
                    icon = L.Icon(icon_url = icons_urls['good'],icon_size=[30, 30]) # good
                elif avg_var >= var_limits[input.var()][0] and avg_var < var_limits[input.var()][1]:
                    icon = L.Icon(icon_url = icons_urls['fair'],icon_size=[30, 30]) # fair
                elif avg_var >= var_limits[input.var()][1] and avg_var < var_limits[input.var()][2]:
                    icon = L.Icon(icon_url = icons_urls['moderate'],icon_size=[30, 30]) # moderate
                elif avg_var >= var_limits[input.var()][2] and avg_var < var_limits[input.var()][3]:
                    icon = L.Icon(icon_url = icons_urls['poor'],icon_size=[30, 30]) # poor
                else:
                    icon = L.Icon(icon_url = icons_urls['very_poor'],icon_size=[30, 30]) # very poor
                
                marker = L.Marker(location=(data_cities_lat_lon[city]['lat'], data_cities_lat_lon[city]['lon']),    # dodanie znacznika
                                icon=icon,
                                draggable=False)
                map.add_layer(marker)

    # When the slider changes, update the map's zoom attribute (2)
    @reactive.Effect
    def _():
        map.zoom = input.zoom()

    # When zooming directly on the map, update the slider's value (2 and 3)
    @reactive.Effect
    def _():
        ui.update_slider("zoom", value=reactive_read(map, "zoom"))

    # funkcja znajdująca miasta najbardziej i najmniej zaniczyszczone wybranym związkiem
    @reactive.Calc
    def worst_best_pol():
        df_time_sec = df_time_section()
        avg_pollution = dict()
        data_cities_lat_lon = reactive_city_lat_lon()

        for city in data_cities_lat_lon:
            df_copy = df_time_sec.loc[df_time_sec['city'] == city]

            start = pd.to_datetime(df_copy['dt'].values[0]) + timedelta(hours = 8)
            end = pd.to_datetime(df_copy['dt'].values[0]) + timedelta(hours = 20)

            df_copy.set_index('dt', inplace=True)
            df_new = df_copy[start:end]

            avg = df_new[input.var()].mean()
            avg_pollution[city] = avg
        
        worst = max(avg_pollution, key=avg_pollution.get)
        best = min(avg_pollution, key=avg_pollution.get)

        return [worst, best]

    # funkcja do wyświetlania informacji
    @output
    @render.ui
    def map_bounds():
        if (input.var() != ''):
            cities = worst_best_pol()
        else:
            cities = ['','']
        return ui.p(f"Highest pollution: {cities[0]}", ui.br(), f"Least pollution: {cities[1]}")

app = App(app_ui, server)
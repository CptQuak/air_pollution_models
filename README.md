Data collected from OWM's API was stored as jsons in data folder, but I don't want to deal with problems of possibly stealing OWM data.


---

## lab-2223-group1

- **Temat.** Utworzenie sieci neuronowej.


## Wymagania biznesowe

- [x] [Wybór zbioru do analizy; zbiór powinien być rzeczywisty.](#zrodla-danych)
- [x] [Zapisanie danych do repozytorium z możliwością aktualizacji tych danych.](#zapis-danych)
    - Aktualizacja danych, które spełniają kontrakt, nie powinna spowodować by analiza się nie wykonała.
- [x] [Przygotowanie danych.](#dane)
    - Decyzja, co zrobić z brakami w danych.
    - Podjęcie decyzji o ewentualnej standaryzacji danych.
    - Decyzja, co zrobić z danymi niezbalansowanym.
    - Decyzja, co zrobić z wartościami silnie odstającymi; interpretacja.
- [x] [Stworzenie modelu sieci w pełni w oparciu o kod.](#trenowanie-modeli)
    - Wybranie odpowiedniej technologii.
    - Opisanie sieci neuronowej kodem.
    - Sprofilowanie sieci neuronowej.
    - Wdrożenie testów jednostkowych.
    - Wprowadzenie testów regresji dla modelu.
    - Sprawdzenie prognozy bądź klasyfikacji.
- [x] [Wdrożenie ciągłej integracji i ciągłego wdrożenia sieci.](#wdrożenie-ciągłej-integracji-i-ciągłego-wdrożenia-sieci)
- [x] [Zapisanie modelu do pliku binarnego.](#zapisanie-modelu-do-pliku-binarnego)
- [x] [Możliwość porównania dwóch modeli.](#możliwość-porównania-dwóch-modeli)


## Postępy prac

### Zrodla danych
- dane pogodowe - https://openweathermap.org/history
- dane o zanieczyszczeniach - https://openweathermap.org/api/air-pollution
- działamy na planie developer, dostep do danych jedne rok wstecz aktualizowanych w 1-godzinnych interwałach - https://openweathermap.org/full-price


### Zapis danych
- [x] Folder z danymi: /data
- [x] Uruchamiając skrypt pobierający dane dla miast z pliku city_list.txt (skrypt - /scripts/get_data.py)
- [x] Dla danych miast tworzone są foldery na zwrócone jsony z API oraz przygotowany jest geocoding potrzebny do pobrania danych (cities_lat_lon_data.json)
- [x] Aktualizacja skryptu do zapisu o możliwość aktualizacji danych, obecna funkcjonalność wyłącznie nadpisuje dane.


### Dane
- [x] Do dalszych analiz obecnie na podstawie pobranych danych przygotowywany jest jednolity plik csv zapisywany do folderu data/csv (skrypt - /scripts/create_csv_files.py)
- [x] Feature engineering - utworzenie cech zwiazanych z sezonowoscia, transformacje zarówno standaryzacja, power transformacje oraz onehotencoding
- [x] Feature selection - obecnie jest wprowadzone alternatywne podejście w postaci PCA
- [x] Dane niezbalansowane
    - Nie dotyczy? 
- [x] Dane odstające
    - Jedyny przypadek danych odstających widzę w przypadku wadliwych czujników, które przesłały nieprawdziwe informacje do bazy OWM. 
- [x] Braki w danych
    - **rain, snow - 1h i 3h**, kolumny generalnie zwracają braki jeśli nie było opadów stąd uzupełnienia zerami.
    - **wind.gust (nie ma w tym linku do api)** - dodali taką nową kolumnę, dosyć niekompletne w zależności od lokalizacji. Interpretacja z tego co zrozumieliśmy to najsilniejszy podmuch wiatru w danym przedziale godzinowym. Ze względu na wysoką niekompletność zdecydowaliśmy się usunąć.
    - **Temp_min, temp_max** - ponieważ przekazywana temperatura jest generalnie średnią z kilku czujników w mieście, wystarczy nam średnia, więc usuwamy te kolumny.
    - Odnośnie air-pollution API, są takie sytuacje, gdzie brakuje danych z całego dnia, w takich przypadkach została zastosowana interpolacja liniowa.


### Trenowanie modeli
- [x] Wybranie odpowiedniej technologii.
      Implementacja w modeli w oparciu o Pytorch
- [x] Opisanie sieci neuronowej kodem.
    - AR/MA/ARIMA, 
    - DNN, 
    - RNN (np LSTM)
- [x] Sprofilowanie sieci neuronowej.
    - Porównanie jednego AR do predykcji wszystkich na raz vs 8 indywidualnych AR 
- [x] Wprowadzenie testów regresji dla modelu.
    - Jeśli performance wybranego najlepszego modelu po cross validacji jest gorszy od poprzedniego modelu to nie zastępujemy
- [x] Sprawdzenie prognozy.
    - Wyświetlenie prognozy na dashboardzie

### Wdrożenie ciągłej integracji i ciągłego wdrożenia sieci.
- [x] Automatyczne pobranie nowych danych, np raz dziennie/co godzine?
- [x] Trenowanie modeli poprzez github actions (obecnie ustawione na wywolanie ręczne)

### Zapisanie modelu do pliku binarnego.
- [x] zapis w postaci plików pickle

### Możliwość porównania dwóch modeli.
- [x] predykcje z roznych modeli są zapisywane do plików csv

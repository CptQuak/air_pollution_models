# Instrukcja pracy ze srodowiskiem w minicondzie

## Instalacja

1. Instalacja minicondy, https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links
	- Po drodze pojawi się opcja Add Miniconda3 to my PATH, zaznacz ją
2. Uruchom miniconde, utwórz nowe środowisko: 
    - conda create -n pzlab python=3.9.6
3. Aktywuj je: 
    - conda activate pzlab
4. Trzeba w minicondzie przejsc do głównego folderu w którym mamy kod projektu, u mnie to jest np komenda: 
    - cd C:\Users\kamil\Documents\Projekt
5. Instalacja wymaganych paczek: 
    - pip install -r requirements.txt
6. Teraz w vscodzie jak uruchamiasz jupytera czy cos ustaw pzlab (pzlab tak nazwałem akurat te środowisko)

## Dodawanie nowych paczek
Jeśli potrzebujemy doinstalować jakąś paczkę, to dodajmy o tym informacje do pliku requirements. Przykład takiego procesu intalacyjnego:

1. Uruchom terminal miniconda3
2. Przejdz do folderu z kodem źródłowym projektu (potrzebne do zaktualizowania requirements), ścieżka może się różnić u mnie np trzeba użyć komendy: 
    - cd C:\Users\kamil\Documents\Projekt
3. Aktywujemy środowisko projektowe: 
    - conda activate pzlab
4. Instalujemy paczkę: 
    - pip install pandas
5. Zapisanie informacji o doinstalowanych paczkach do pliku, pip freeze da informacje o obecnie zainstalowanych paczkach. 
6. Znajdź tą główną nazwę paczki i dodają o niej informację z wersją i dopisz do requirements.txt np.:
    - torch==2.0.0
6. Zrobione, zrób commita z informacją o aktualizacji requirements


## Doinstalowanie brakujących paczek z requirements.txt
1. Otwórz terminal miniconda3
2. Aktywuj środowisko: conda activate pzlab
3. Przejdz do folderu projektu: cd C:\Users\kamil\Documents\Projekt
4. Zinstaluj nowe paczki prosto z pliku: pip install -r requirements.txt


## Dostep do api
1. Make sure https://pypi.org/project/python-dotenv/ is installed
2. Create .env file in the root folder with an api key inside,

Example:
owm_api_key = KEYNUMBERS123123123
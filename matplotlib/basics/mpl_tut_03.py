"""
Simple reversed bar plot using data from .csv
"""

import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

# Pobieramy dane z pliku uzywajac funkcji slownika (klucz -> wartosc)
# with open('data.csv') as csv_file:
#     csv_reader = csv.DictReader(csv_file)

#     # Wyswietlimy z czym pracujemy (1 wiersz)
#     row = next(csv_reader)
#     # Uzywamy klucza -> dostajemy wartosci pod tym kluczem
#     # Dodatkowo splitujemy zeby dostac liste
#     print(row["LanguagesWorkedWith"].split(";"))

#     # Wiemy z czym pracujemy, teraz uzyjemy funkcji counter do zliczania ile mamy poszczegolnych wystapien danych jezykow w calej liscie
#     # Counter zlicza wystapeinia w liscie, dziala jak slownik
#     language_counter = Counter()

#     for row in csv_reader:
#         language_counter.update(row["LanguagesWorkedWith"].split(";"))

# Inna metoda pobrania danych, przy pomocy pandas, jest szybsza
data = pd.read_csv("data-03.csv")
ids = data["Responder_id"]
lang_responses = data["LanguagesWorkedWith"]
language_counter = Counter()

for response in lang_responses:
    language_counter.update(response.split(";"))

# print(language_counter)
# Chcemy tylko 15 najpopularniejszych
# print(language_counter.most_common(15))

# Rozpakujemy nasz slownik language_counter do list
popularity, language = [], []

for item in language_counter.most_common(15):
    language.append(item[0])
    popularity.append(item[1])


# Najpopularniejsze chcemy na samej gorze
popularity.reverse()
language.reverse()

# Wykres slupkowy
plt.style.use("bmh")
# barh to wykres w poziomie, bar to standardowy w pionie
plt.barh(language, popularity)

plt.title("Most popular languages")
plt.xlabel("People using this")
plt.show()

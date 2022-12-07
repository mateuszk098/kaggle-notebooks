"""
Creating subplots.
"""

import pandas as pd
from matplotlib import pyplot as plt
import sys
import json
with open('lang.json', 'r', encoding='utf-8') as lang_file:
    lang = json.load(lang_file)

while True:
    lang_choice = input('Language (EN/PL): ')
    if lang_choice == 'EN' or lang_choice == 'PL':
        break
    else:
        print("ValueError")


plt.style.use("bmh")

data = pd.read_csv("data-11.csv")
ages = data["Age"]
dev_salaries = data["All_Devs"]
py_salaries = data["Python"]
js_salaries = data["JavaScript"]

# Tworzymy subplot
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

ax1.plot(ages, py_salaries, label="Python")
ax1.plot(ages, js_salaries, label="JavaScript")
ax1.set_ylabel(lang[lang_choice]["Median Salary (USD)"])
ax1.set_title(lang[lang_choice]["Median Salary (USD) by Age"])
ax1.legend(loc="upper left")

ax2.plot(ages, dev_salaries,
         label=lang[lang_choice]["All Developers"], linestyle="--")
ax2.set_xlabel(lang[lang_choice]["Age"])
ax2.set_ylabel(lang[lang_choice]["Median Salary (USD)"])
ax2.legend(loc="upper left")

plt.show()

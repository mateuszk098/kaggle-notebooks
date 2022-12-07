"""
Simple bar plot with legend.
"""

import matplotlib.pyplot as plt
import numpy as np

# Median Developer Salaries by Age
ages = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

# Median Developer Salaries by Age
dev = [38496, 42000, 46752, 49320, 53200,
       56000, 62316, 64928, 67317, 68748, 73752]

# Median Python Developer Salaries by Age
py_dev = [45372, 48876, 53850, 57287, 63016,
          65998, 70003, 70000, 71496, 75370, 83640]

# Median JavaScript Developer Salaries by Age
js_dev = [37810, 43515, 46823, 49293, 53437,
          56373, 62375, 66674, 68745, 68746, 74583]


plt.style.use("bmh")
# Indeksy sa potrzebne do osi x, potem zmienimy oznaczenia na osi x
x_indexes = np.arange(len(ages))
# Rozstawienie slupkow
width = 0.25

# Tutaj width = 0.2 to szerokosc slupka, rozstawienie jest szersze zeby slupki nie dotykaly sie
plt.bar(x_indexes - width, dev, width=0.2, label="All devs")
plt.bar(x_indexes, py_dev, width=0.2, color="blue", label="Python devs")
plt.bar(x_indexes + width, js_dev, width=0.2,
        color="orange", label="JavaScript devs")
plt.xlabel("Age")
plt.ylabel("Devs salaries (USD)")
plt.title("Developers salaries by age")

# Zmieniamy nazwy osi x na wiek
plt.xticks(ticks=x_indexes, labels=ages)

plt.legend()
plt.show()

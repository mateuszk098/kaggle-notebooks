"""
Real-time plot. The `data_gen.py` must be launched for this. 
"""

import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


plt.style.use("bmh")

# x_vals = []
# y_vals = []

# index = count()


# def animate(i):
#     x_vals.append(next(index))
#     y_vals.append(random.randint(0, 5))

#     plt.cla()
#     plt.plot(x_vals, y_vals)

# Program data_gen.py generuje w czasie rzeczywistym wartosic losowe z trendem, ktore sa zapisywane do pliku csv, z tego samego pliku program (ten) odczytuje wartosci i umieszcza na wykresie
def animate(i):
    data = pd.read_csv("data-10.csv")
    x = data["x_value"]
    y1 = data["total_1"]
    y2 = data["total_2"]

    plt.cla()
    plt.plot(x, y1, label="Channel 1")
    plt.plot(x, y2, label="Channel 2")

    plt.legend(loc="upper left")


# plt.gcf() to get current figure, wywolujemy obecny wykres oraz funkcje ktora losuje liczby z interwalem czasowym interval w ms
ani = FuncAnimation(plt.gcf(), animate, interval=1000)
plt.show()

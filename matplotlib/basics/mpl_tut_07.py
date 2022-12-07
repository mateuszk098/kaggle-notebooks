"""
Creating histogram.
"""

import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("data-07.csv")
ids = data["Responder_id"]
ages = data["Age"]
median_age = int(ages.median())

bins = [20, 23, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59]

plt.style.use("bmh")

plt.hist(ages, bins=bins, edgecolor="black")
plt.axvline(median_age, color="orange", label="Median Age")
plt.xticks(ticks=bins, labels=bins)

plt.xlabel("Age")
plt.ylabel("Number of Responders")
plt.legend(loc="upper right")
plt.title("Age of Responders")
plt.show()

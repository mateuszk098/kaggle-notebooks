"""
Filling area between linear plots.
"""

from numpy import where
import pandas as pd
from matplotlib import pyplot as plt

# Czytamy dane
data = pd.read_csv("data-06.csv")
ages = data["Age"]
dev_salaries = data["All_Devs"]
py_salaries = data["Python"]
js_salaries = data["JavaScript"]

plt.style.use("bmh")

plt.plot(ages, js_salaries, color="black", linewidth=1.5,
         linestyle="--", label="JavaScript")
plt.plot(ages, py_salaries, color="blue", linewidth=1.5, label="Python")

overall_median = 58000

# Wypelnia obszar pod krzywa pythona dla wartosci wiekszych od overall_median i nad krzywa dla wartosci mniejszych od overall_median, alpha to stopien zaciemnienia

# plt.fill_between(ages, py_salaries, overall_median, where=(
#     py_salaries > overall_median), interpolate=True, alpha=0.25)

# plt.fill_between(ages, py_salaries, overall_median, where=(
#     py_salaries <= overall_median), interpolate=True, alpha=0.25)

plt.fill_between(ages, py_salaries, js_salaries, where=(
    py_salaries > js_salaries), interpolate=True, alpha=0.25, label="Above JS")

plt.fill_between(ages, py_salaries, js_salaries, where=(
    py_salaries <= js_salaries), interpolate=True, alpha=0.25, label="Below JS")


plt.xlabel("Age")
plt.ylabel("Median Salary (USD)")
plt.title("Developers Median Salary (USD)", fontname="Times New Roman")
plt.legend(loc="upper left")

plt.show()

"""
Simple pie chart.
"""

from matplotlib import pyplot as plt

# Language Popularity
slices = [59219, 55466, 47544, 36443, 35917]
labels = ['JavaScript', 'HTML/CSS', 'SQL', 'Python', 'Java']
explode = [0.05, 0.05, 0.05, 0.05, 0.05]

plt.style.use("bmh")

# W pie chart nie mamy osi, wiec wszystkie dane zawieraja sie w slices, wedgeprops to slownik, tutaj opcja edgecolor to obramowanie
# Explode to wyciecia miedzy slice
# startangle usatwia poczatkowy kat, shadow to lekki cien, autopct to procenty wewnatrz
plt.pie(slices, labels=labels, explode=explode, startangle=90, shadow=True,
        autopct="%1.1f%%", wedgeprops={"edgecolor": "black"})

plt.title("5 most common languages according to stackoverflow")
plt.show()

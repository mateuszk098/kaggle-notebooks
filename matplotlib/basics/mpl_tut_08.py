"""
Simple scatter plot.
"""

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
y = (2, 4, 6, 8, 10, 12, 14, 16, 14, 12, 10, 8)
colors = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
sizes = (50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105)

plt.style.use("bmh")
# Rozmiar
figure(figsize=(8, 4.5), dpi=100)

# s to rozmiar punktu, edgecolor to kolor krawedzi punktu, alpha zmienia stopien zaciemnienia koloru, linewidth to grubosc krawedzi punktu
# plt.scatter(x, y, s=100, c="green", marker="X",
#             edgecolor="black", alpha=0.75, linewidth=1)

# to wykres z kolorem odpowiadajacym danemu punktowi, powiedzmy ze kolor to stopien satyfkacji np klienta, cmap to mapa koloru - odcienie zielone, sizes zmienia sie wraz z kolejnymi punktami
# plt.scatter(x, y, s=sizes, c=colors, cmap="Greens", marker="o",
#             edgecolor="black", alpha=0.75, linewidth=1)
# cbar = plt.colorbar()
# cbar.set_label("Satisfaction")


# Czytamy realne dane, to liczba wyswietlen, like, procent likow do unlikow na yt w danym dniu na top liscie
data = pd.read_csv("data-08.csv")
view_count = data["view_count"]
likes = data["likes"]
ratio = data["ratio"]

# Wykres (x,y) = (views, likes) gdzie dodatkowo dla kazdego filmu stosujemy odpowiedni kolor ktory mowi o stosunku (%) likow do unlikow
plt.scatter(view_count, likes, c=ratio, cmap="summer", marker="o",
            edgecolor="black", alpha=0.75, linewidth=1)
plt.xscale("log")
plt.yscale("log")
cbar = plt.colorbar()
cbar.set_label("Like/Dislike Ratio")

plt.xlabel("View Count", fontname="Times New Roman")
plt.ylabel("Total Likes", fontname="Times New Roman")
plt.title("Trending YouTube Videos", fontname="Times New Roman")
plt.show()

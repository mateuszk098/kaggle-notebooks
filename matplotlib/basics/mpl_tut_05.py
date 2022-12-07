"""
Simple stack plot.
"""

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

minutes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
player_1 = [1, 2, 3, 3, 3, 4, 4, 5, 5]
player_2 = [1, 1, 2, 2, 3, 3, 3, 4, 4]
player_3 = [1, 1, 1, 2, 2, 2, 2, 3, 3]

labels = ["Player 1", "Player 2", "Player 3"]

plt.style.use("bmh")

# Rozmiar
figure(figsize=(7, 4), dpi=100)

plt.stackplot(minutes, player_1, player_2, player_3, labels=labels)
plt.title("My Stackplot")
plt.xlabel("Minute")
plt.ylabel("Player Score")
plt.legend(loc="upper left")

plt.show()

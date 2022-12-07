"""
Time plot.
"""

import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

plt.style.use("bmh")

# dates = [datetime(2021, 9, 14), datetime(2021, 9, 15), datetime(
#     2021, 9, 16), datetime(2021, 9, 17), datetime(2021, 9, 18), datetime(2021, 9, 19)]
# values = [1, 2, 4, 6, 2, 4]

# plt.plot_date(dates, values, linestyle="solid")
# # Oznaczenia daty osi x pod katem
# plt.gcf().autofmt_xdate()
# # Format daty - nazwa miesiaca, dzien rok
# date_format = mpl_dates.DateFormatter("%b, %d %Y")
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.show()

# Dane z csv
data = pd.read_csv("data-09.csv")
# Zamieniamy string na daty
data["Date"] = pd.to_datetime(data["Date"])
# Sortujemy daty gdyby dni były nie po kolei
data.sort_values("Date", inplace=True)

price_date = data["Date"]
price_close = data["Close"]

plt.plot_date(price_date, price_close, linestyle="solid")
plt.gcf().autofmt_xdate()

plt.title("Bitcoin Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

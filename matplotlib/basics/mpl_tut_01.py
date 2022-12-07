"""
Simple linear plot with legend.
"""

import matplotlib.pyplot as plt


def draw_plot(ages, dev, py_dev, js_dev):

    # print(plt.style.available)

    plt.style.use("bmh")

    plt.plot(ages, dev, color="orange", marker=".", label="All Devs")
    plt.plot(ages, py_dev, color="blue", marker=".",  label="Python Devs")
    plt.plot(ages, js_dev, color="green", marker=".", label="JavaScript Devs")

    plt.xlabel("Devs Age")
    plt.ylabel("Median Salary (USD)")
    plt.title("Median Salary (USD) by Age")
    plt.legend(loc="upper left")
    plt.tight_layout()

    # plt.savefig("devs.png")
    plt.show()


if __name__ == "__main__":
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

    draw_plot(ages, dev, py_dev, js_dev)

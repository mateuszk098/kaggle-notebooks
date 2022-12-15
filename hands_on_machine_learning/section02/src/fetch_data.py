"""
Mini module to make a separate directory, download necessary data and unzip it.
"""

import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
DATA_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
IMG_URL = DOWNLOAD_ROOT + "images/end_to_end_project/california.png"


def fetch_housing_data(data_url=DATA_URL):
    """Create a new 'data' directory, download the `data_url`
    file to this directory and finally unzip it."""

    data_path = os.path.join("..", "data")
    os.makedirs(data_path, exist_ok=True)
    tgz_path = os.path.join(data_path, "housing.tgz")
    urllib.request.urlretrieve(data_url, tgz_path)

    with tarfile.open(tgz_path) as tf:
        tf.extractall(path=data_path)


def fetch_california_map(img_url=IMG_URL):
    """Create a new 'img' directory and download the `img_url` file to this directory."""

    img_path = os.path.join("..", "img")
    os.makedirs(img_path, exist_ok=True)
    california_path = os.path.join(img_path, "california.png")
    urllib.request.urlretrieve(img_url, california_path)


if __name__ == "__main__":
    fetch_housing_data()
    fetch_california_map()

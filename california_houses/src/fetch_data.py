"""
Mini module to make a separate directory, download necessary data and unzip it.
"""

import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
DATA_PATH = os.path.join("..", "data")
DATA_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(data_url=DATA_URL, data_path=DATA_PATH):
    """Create a new directory in `data_path`, download the `data_url`
    file to this directory and finally unzip it."""

    os.makedirs(data_path, exist_ok=True)
    tgz_path = os.path.join(data_path, "housing.tgz")
    urllib.request.urlretrieve(data_url, tgz_path)

    with tarfile.open(tgz_path) as tf:
        tf.extractall(path=data_path)


if __name__ == "__main__":
    fetch_housing_data()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:07:06 2018

@author: joans

Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/, store it into
a dictionary and save it in pickle format. Then, load() reads and returns this
dictionary.

Adapted from https://github.com/hsjeong5/MNIST-for-Numpy
"""

import numpy as np
from urllib import request
import gzip
import pickle
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('Logger')

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

folder = "MNISTData"


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    try:
        os.mkdir(folder)
        for name in filename:
            print("Downloading " + name[1] + "...")
            request.urlretrieve(base_url + name[1], name[1])
            shutil.move(name[1], folder)
        logger.info("Download complete.")
    except FileExistsError:
        logger.info("files have already been downloaded before.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(folder + "/" + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(folder + "/" + name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(folder + "/" + "mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    logger.info("Save complete.")


def download_and_save():
    download_mnist()
    save_mnist()


def load():
    with open(folder + "/" + "mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


if __name__ == '__main__':
    download_and_save()

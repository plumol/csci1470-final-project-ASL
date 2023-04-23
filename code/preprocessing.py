import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import pathlib



train_dir = r"data/handgesturedataset_part1"

# TODO: load the dataset into arrays

# TODO: preprocess the images into a standard size and change to grayscale

# TODO: shuffle the arrays for training and test split


def preprocess2(file_path):
    for image in os.listdir(file_path):
        loaded_image = tf.keras.preprocessing.image.load_img(file_path + "/" + image)
        print(loaded_image)

#preprocess(train_dir)
img_dict = preprocess2(train_dir)

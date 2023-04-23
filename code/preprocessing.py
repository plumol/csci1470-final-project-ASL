import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import pathlib



train_dir = r"data/handgesturedataset_part1"


def preprocess(file_path):
    input_images = []
    # For showing images.
    pil = []
    # Gets 0 through 36, repeats each of them 25x. 
    labels = np.repeat(np.arange(36), 25) 
    # Output of length 25.
    labels = tf.one_hot(labels, 36) 

    for image in os.listdir(file_path):
        loaded_image = tf.keras.preprocessing.image.load_img(file_path + "/" + image, target_size=(28, 28), grayscale=True)
        pil.append(loaded_image)
        loaded_image = tf.keras.preprocessing.image.img_to_array(loaded_image) / 255.
        input_images.append(loaded_image)

    # indices = [x for x in range(len(images))]
    # indices = tf.random.shuffle(indices)
    # images = tf.gather(images, indices)
    # labels = tf.gather(labels, indices)

    return input_images, pil, labels


imgs, pils, labels = preprocess(train_dir)

# pils[<index>].show()

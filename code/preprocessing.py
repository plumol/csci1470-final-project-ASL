import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os
import tensorflow as tf
import string


# About this data set: contains 36 different ASL hand signs: 0-9, A-Z with 25 images of each sign
train_dir = r"data/handgesturedataset_part1"

def preprocess(file_path):
    """
    Given a file path for the dataset, we will take the images and transform it into a format our
    model can use. We fine-tuned this preprocessing method according to the paper we choose, with some additional
    features. The images in our ASL dataset are reshaped to a (28, 28) image size and grayscaled. These pixel 
    values are then normalized by 255. 

    Since we are adapting a dataset from another paper, it doesn't necessarily have its own labels. To solve this,
    we developed a numerical labeling system for each of the 36 available ASL signs. Each sign is assigned a number 
    and each of their images in the dataset are labeled accordingly. Since the dataset is ordered by sets of 25 images,
    we can iterate over each. 

    :return: input_images: A list containing our preprocessed input images
    pil: This list contains PIL formatted images that haven't been converted to an array yet, which allows for quick processing.
    labels: A list with labels for the input_images, created as described above. 
    """
    input_images = []
    # For showing images directly. You can use the PIL list at the end to 
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


    return input_images, pil, labels

def split_train_test(input_images, input_labels, train_split = .80):
    # TODO: find a way to do an efficient train test split, since our labeled data is in order, we need to split some of the labeled images
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # splits processed data set into respective collections of signs: 25 images for each of 36 signs
    # So we have 36 sublists of 25 images
    split_input_images = np.array_split(ary=input_images, indices_or_sections=36)
    split_input_labels = np.array_split(ary=input_labels, indices_or_sections=36)

    # shuffle each image:label collection, similar to how we shuffled the training data
    indices = [x for x in range(len(split_input_images[0]))]
    indices = tf.random.shuffle(indices)

    # gather the shuffled iamges and indices 
    for i in range(len(split_input_images)):
        split_input_images[i] = tf.gather(split_input_images[i], indices)
        split_input_labels[i] = tf.gather(split_input_labels[i], indices)


    # after shuffling, split each into a train, test collection
    
    for i in range(len(split_input_images)):
        # since we have a shuffled array of images, take the first 80% as training and last 20% as testing
        train_image_chunk = split_input_images[i][:int(len(split_input_images[0]) * train_split)]
        train_label_chunk  = split_input_labels[i][:int(len(split_input_labels[0]) * train_split)]

        test_image_chunk = split_input_images[i][int(len(split_input_images[0]) * train_split):]
        test_label_chunk = split_input_labels[i][int(len(split_input_labels[0]) * train_split):]

        train_images.append(train_image_chunk)
        train_labels.append(train_label_chunk)
        test_images.append(test_image_chunk)
        test_labels.append(test_label_chunk)

    # Since we have 36 sublists, we need to concatenate them to reform one single array
    train_images = np.concatenate(train_images)
    train_labels = np.concatenate(train_labels)
    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    return train_images, train_labels, test_images, test_labels

def label_name_dict():
    """
    Instead of using the numbers we generated for the labels, we can have a dictionary that interprets the 
    predicted label into the corresponding sign. 
    """
    # lazy- I didn't want to write out each dict entry for the labels, so I automated it
    nums_alpha = np.arange(0, 10)
    nums_alpha = np.concatenate([nums_alpha, list(string.ascii_lowercase)])
    
    label_name = {
    }
    for i in range(36):
        label_name[i] = nums_alpha[i]

    return label_name


imgs, pils, labels = preprocess(train_dir)
split_train_test(imgs, labels)

label_name_dict()
# pils[<index>].show()


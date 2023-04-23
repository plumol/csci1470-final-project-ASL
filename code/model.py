import numpy as np
import math
import tensorflow as tf
from keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
import sklearn

class ASLClassifier(tf.keras.Model):
    def __init__(self):
        super(ASLClassifier, self).__init__()
        self.num_epochs = 30
        self.num_classes = 36
        self.batch_size = 120
        self.learning_rate = 0.01

        self.layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3)),
            tf.keras.layers.Conv2D(64, (3,3)),
            tf.keras.layers.MaxPool2D(strides=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(36)
        ])


    def call(self, x):
        return self.layers(x)

    def compile():
    
    def train():

    def test():
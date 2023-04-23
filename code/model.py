import numpy as np
import math
import tensorflow as tf
import tensorflow_ranking as tfr
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
        self.loss_function = tfr.keras.losses.SoftmaxLoss()
        self.acc_function = tf.keras.metrics.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

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
    
    def train(self):
        total_loss = 0
        for _ in self.num_epochs:

            with tf.GradientTape() as tape:
                loss = self.loss_function()
                acc = self.acc_function()
            
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            ## Compute and report on aggregated statistics
            total_loss += loss # do we actually want total loss? maybe avg loss? also want acc

        return total_loss

    def test():
        # should be similar to training

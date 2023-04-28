import numpy as np
import math
import tensorflow as tf
from keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense
import sklearn

class ASLClassifier(tf.keras.Model):
    def __init__(self):
        super(ASLClassifier, self).__init__()
        self.num_epochs = 20
        self.num_classes = 36
        self.batch_size = 120
        self.learning_rate = 0.01
        #self.loss_function = tfr.keras.losses.SoftmaxLoss()
        #self.acc_function = tf.keras.metrics.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.classify = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3)),
            tf.keras.layers.Conv2D(64, (3,3)),
            tf.keras.layers.MaxPool2D(strides=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(36) # number of classes/hand signs
        ])


    def call(self, x):
        return self.classify(x)
    
    def train(self, train_inputs, train_labels, isTesting = False):
        #TODO add accuracy

        indices = [x for x in range(len(train_inputs))]
        indices = tf.random.shuffle(indices)
        train_inputs = tf.gather(train_inputs, indices)
        train_labels = tf.gather(train_labels, indices)


        total_loss = []
        total_acc = []
        for b, b1 in enumerate(range(self.batch_size, train_inputs.shape[0] + 1, self.batch_size)):
            b0 = b1 - self.batch_size
            with tf.GradientTape() as tape:
                pred_labels = self.call(train_inputs[b0:b1])

                loss = self.loss_function(pred_labels, train_labels[b0:b1])
                acc = self.acc_function(pred_labels, train_labels[b0:b1])
            
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            ## Compute and report on aggregated statistics
            total_loss.append(loss) # do we actually want total loss? maybe avg loss? also want acc
            total_acc.append(acc)

        return tf.reduce_mean(total_loss), tf.reduce_mean(total_acc)

    def test(self, test_inputs, test_labels):
        # should be similar to training
        #HW3:
        # logits = model.call(test_inputs, True)
        # return model.accuracy(logits, test_labels)

        total_loss = []
        total_acc = []
        for b, b1 in enumerate(range(self.batch_size, len(test_labels)+1, self.batch_size)):
            
            b0 = b1 - self.batch_size
            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            
            probs = self.call(test_inputs[b0:b1])
            loss = self.loss_function(probs, test_labels[b0:b1])
            acc = self.acc_function(probs, test_labels[b0:b1])

            total_loss.append(loss)
            total_acc.append(acc)


        return tf.reduce_mean(total_loss), tf.reduce_mean(total_acc)
        pass

    def loss_function(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def acc_function(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

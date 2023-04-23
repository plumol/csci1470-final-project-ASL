import tensorflow as tf
import argparse
from model import ASLClassifier

def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           required=True,              choices=['train', 'test', 'both'],  help='Task to run')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def train_classifier(model):
    """
    Train your classifier with one epoch

    Returns:
    loss? accuracy?
    """
    total_loss = 0
    optimizer = tf.optimizers.Adam(model.learning_rate)
    # for data, labels in train_loader:

    #     with tf.GradientTape() as tape:
    #         if is_cvae:
    #             oh_labels = one_hot(labels, model.num_classes)
    #             # data_casted = tf.cast(data, dtype=tf.float32)
    #             # labels_casted = tf.cast(oh_labels, dtype=tf.float32)
    #             x_hat, mu, logvar = model(data, oh_labels)
    #         else: 
    #             x_hat, mu, logvar = model(data)
    #         loss = loss_function(x_hat, data, mu, logvar)
        
    #     grads = tape.gradient(loss, model.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #     ## Compute and report on aggregated statistics
    #     total_loss += loss

    # return total_loss
    # ^ TAKEN FROM HW5 - USE AS INSPIRATION

def main(args):
    # compile model
    # train model
    # test model
    pass

if __name__ == '__main__':
    main(parse_args())
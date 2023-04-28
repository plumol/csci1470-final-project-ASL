import random
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import argparse
from model import ASLClassifier as model

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

def train_classifier(model, train_inputs, train_labels): # is this the right approach??
    """
    Train your classifier with one epoch

    Returns:
    loss? accuracy?
    """
    # copied from hw3?
    zipped = list(zip(train_inputs, train_labels))
    random.shuffle(zipped)
    inputs = np.array([tf.image.random_flip_left_right(tup[0]) for tup in zipped])
    labels = np.array([tup[1] for tup in zipped])
 
    for i in range(0, len(zipped), model.batch_size):
        with tf.GradientTape() as tape:
            logits = model.call(inputs[i:i + model.batch_size], False)
            loss = model.loss(logits, labels[i:i + model.batch_size])

        model.loss_list.append(loss)
        grads = tape.gradient(loss, model.trainable_variables) # calculates gradients
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables)) # updates trainable vars
    # total_loss = 0
    # optimizer = tf.optimizers.Adam(model.learning_rate)
    # '''Trains model and returns model statistics'''
    # stats = []
    # try:
    #     for epoch in range(model.epochs):
    #         stats += [model.train(captions, img_feats, pad_idx, batch_size=args.batch_size)]
    #         if args.check_valid:
    #             model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
    # except KeyboardInterrupt as e:
    #     if epoch > 1:
    #         print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
    #     else: 
    #         raise e
        
    # return stats
    # ^ TAKEN FROM HW5 - USE AS INSPIRATION

def compile_model(model):
    '''Compiles model by reference based on arguments'''
    # optimizer = tf.keras.optimizers.get(args.optimizer).__class__(learning_rate = args.lr)
    model.compile(
        optimizer   = model.optimizer, # play around with this
        loss        = model.loss_function, # reduce mean?
        metrics     = [model.acc_function] # not sure if cat crossentropy is what we want lol
    )

def test_model(model, captions, img_feats, pad_idx, args):
    '''Tests model and returns model statistics'''
    perplexity, accuracy = model.test(captions, img_feats, pad_idx, batch_size=args.batch_size)
    return perplexity, accuracy

def main(args):
    compile_model(model)

    pass

if __name__ == '__main__':
    main(parse_args())
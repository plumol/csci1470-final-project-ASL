import random
import numpy as np
import tensorflow as tf
import argparse
from model import ASLClassifier as model
from preprocessing import preprocess, split_train_test, label_name_dict
import matplotlib.pyplot as plt
import math


def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           required=False ,              choices=['train', 'test', 'both'],  help='Task to run') # change required back to true
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
 
    for epoch in range(model.num_epochs):
        avg_loss, avg_acc = model.train(train_inputs, train_labels)
        print(f"Train epoch: {epoch} \t Loss:{avg_loss} \t Acc:{avg_acc}")

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
        optimizer   = tf.keras.optimizers.Adam(learning_rate = 0.001), # play around with this
        loss        = model.loss_function, # reduce mean?
        metrics     = [model.acc_function] # not sure if cat crossentropy is what we want lol
    )

def test_model(model, test_inputs, test_labels):
    '''Tests model and returns model statistics'''
    loss, accuracy = model.test(test_inputs, test_labels)
    return loss, accuracy


#to visualize, set up a dictionary for all of the generated labels, pull from that
# remember reverse one-hot encoding from hw 2?
# stolen from hw3
def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns

    

    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()

def visualize_inputs(train_images, train_labels, label_names):
    
    #plt.figure(figsize=(10, 10))

    pass

def main(args):
    train_dir = r"data/handgesturedataset_part1"
    imgs, pils, labels = preprocess(train_dir)
    train_images, train_labels, test_images, test_labels = split_train_test(input_images=imgs, input_labels=labels)
    
    label_name = label_name_dict()
    visualize_inputs(train_images=train_images, train_labels=train_labels, label_names=label_name)
    
    
    asl_model = model()
    compile_model(asl_model)
    train_classifier(model=asl_model, train_inputs=train_images, train_labels=train_labels)
    
    test_loss, test_accuracy = test_model(model=asl_model, test_inputs=test_images, test_labels=test_labels)
    print(f"Testing loss: {test_loss}, \t Testing acc: {test_accuracy}")

    visualize_results(test_images[0:50], asl_model.call(test_images), test_labels[0:50], 4, 1)
    pass

if __name__ == '__main__':
    main(parse_args())


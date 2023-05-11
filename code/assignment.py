import random
import numpy as np
import tensorflow as tf
import argparse
from model import ASLClassifier as model
from model import loss_function, acc_function 
from contour_real_time import run_contour_real_time
from hand_detector_real_time import run_hd_real_time
from preprocessing import preprocess, split_train_test, label_name_dict
import matplotlib.pyplot as plt
import math

label_name = label_name_dict()

def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           required=False ,              choices=['train', 'test', 'both'],  help='Task to run')
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def train_classifier(model, train_inputs, train_labels):
    """
    Train the classifier with one epoch
    """
    
    zipped = list(zip(train_inputs, train_labels))
    random.shuffle(zipped)
    # Flip images so that model is slightly more robust
    inputs = np.array([tf.image.random_flip_left_right(tup[0]) for tup in zipped])
    labels = np.array([tup[1] for tup in zipped])
 
    for epoch in range(model.num_epochs):
        avg_loss, avg_acc = model.train(train_inputs, train_labels)
        print(f"Train epoch: {epoch} \t Loss:{avg_loss} \t Acc:{avg_acc}")


def compile_model(model):
    '''Compiles model by reference based on arguments'''
    model.compile(
        optimizer   = tf.keras.optimizers.Adam(learning_rate = 0.001), 
        loss        = loss_function,
        metrics     = [acc_function]
    )

def test_model(model, test_inputs, test_labels):
    '''Tests model and returns model statistics
    Returns: loss and accuracy metrics'''
    loss, accuracy = model.test(test_inputs, test_labels)
    return loss, accuracy

def save_model(model):
    tf.keras.models.save_model(model, r"model")
    print(f"Model saved to /model")

def load_model(model_path):
    loaded_model = tf.keras.models.load_model(model_path,
                                       custom_objects = dict(
                                           loss_function = loss_function,
                                           acc_function = acc_function
                                        ))
    compile_model(loaded_model)
    return loaded_model

# Stolen from hw3
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

def show_incorrect_predictions(prediction, test_labels):
    num_list = [np.argmax(i) for i in prediction]
    pred_list = [label_name[i] for i in num_list]

    incorrect_list = []
    for i, j in zip(enumerate(pred_list), test_labels):
        if i[1] != label_name[np.argmax(j)]:
            incorrect_list.append((i[0], label_name[np.argmax(j)]))

    
    for pred in incorrect_list:
        incorrect = pred_list[pred[0]]
        print(f"Pred: {incorrect}, when actual: {pred[1]}")

def main(args):
    train_dir = r"data/handgesturedataset_part1"
    imgs, pils, labels = preprocess(train_dir)
    train_images, train_labels, test_images, test_labels = split_train_test(input_images=imgs, input_labels=labels)
    
    # visualize_inputs(train_images=train_images, train_labels=train_labels, label_names=label_name)
    
    asl_model = model()
    compile_model(asl_model)
    train_classifier(model=asl_model, train_inputs=train_images, train_labels=train_labels)
    
    test_loss, test_accuracy = test_model(model=asl_model, test_inputs=test_images, test_labels=test_labels)
    print(f"Testing loss: {test_loss}, \t Testing acc: {test_accuracy}")
    # Uncomment the line below if you want to try the contour-based hand detection
    # run_contour_real_time(asl_model, label_name)
    run_hd_real_time(asl_model, label_name)

    pass

if __name__ == '__main__':
    main(parse_args())


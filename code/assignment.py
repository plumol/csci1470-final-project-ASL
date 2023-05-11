import numpy as np
import tensorflow as tf
from model import ASLClassifier as model
from model import loss_function, acc_function 
from contour_real_time import run_contour_real_time
from hand_detector_real_time import run_hd_real_time
from preprocessing import preprocess, split_train_test, label_name_dict

label_name = label_name_dict()

def train_classifier(model, train_inputs, train_labels): 
    """
    Train your classifier with all epochs

    """
    
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

def show_incorrect_predictions(prediction, test_labels):
    num_list = [np.argmax(i) for i in prediction] # find the argmax (decision) of the predictions
    pred_list = [label_name[i] for i in num_list] # translate each decision to the appropriate label in label_name

    # checks if the predicted labels match with the test labels and add to an incorrect list
    incorrect_list = []
    for i, j in zip(enumerate(pred_list), test_labels):
        if i[1] != label_name[np.argmax(j)]:
            incorrect_list.append((i[0], label_name[np.argmax(j)]))

    # print out the incorrect list in a nice way
    for pred in incorrect_list:
        incorrect = pred_list[pred[0]]
        print(f"Pred: {incorrect}, when actual: {pred[1]}")

def main():
    train_dir = r"data/handgesturedataset_part1"
    imgs, pils, labels = preprocess(train_dir)
    train_images, train_labels, test_images, test_labels = split_train_test(input_images=imgs, input_labels=labels)    
    
    asl_model = model()
    compile_model(asl_model)
    train_classifier(model=asl_model, train_inputs=train_images, train_labels=train_labels)
    
    test_loss, test_accuracy = test_model(model=asl_model, test_inputs=test_images, test_labels=test_labels)
    print(f"Testing loss: {test_loss}, \t Testing acc: {test_accuracy}")
    # Uncomment the line below if you want to try the contour-based hand detection
    # run_contour_real_time(asl_model, label_name)
    run_hd_real_time(asl_model, label_name)

    prediction = asl_model.predict(test_images[:])
    save_model(asl_model)

    show_incorrect_predictions(prediction=prediction, test_labels=test_labels)
    

if __name__ == '__main__':
    main()
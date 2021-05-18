import os
import argparse
from pathlib import Path

# teaching utils
import pandas as pd
import numpy as np
from utils.neuralnetwork import NeuralNetwork # from class

# sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from utils.load_mnist import load_mnist # own util


def main(data_path, epochs):
    outpath = 'output'

    # Load data as np arrays
    img, label = load_mnist(data_path)

    # we are assuming the min and max values for pixel intensities are between 0 and 255
    img = img / 255.0 # normalize pixel vals to between 0 and 1 as float

    # get number of unique classes (10) to use as output layer size
    classes = set(label)
    num_classes = len(classes)

    # split our data 80/20 - train/test
    img_train, img_test, label_train, label_test = train_test_split(img, label, random_state=1337, test_size=0.2)

    # convert labels to binary representation (e.g. 2 becomes [0,0,1,0,0,0,0,0,0,0])
    label_train = LabelBinarizer().fit_transform(label_train)
    label_test = LabelBinarizer().fit_transform(label_test)

    # specify the neural network structure with 2 small hidden layers
    neural_network = NeuralNetwork([img_train.shape[1], 32, 16, num_classes]) # 1 input node for every pixel in images, 1 output node for every class

    # train the model
    neural_network.fit(img_train, label_train, epochs=epochs) # prints loss every 100 epochs

    # make predictions on all test images
    label_pred = neural_network.predict(img_test)
    label_pred = label_pred.argmax(axis=1) # give us the highest probability label

    # generate comparative metrics with test data
    classifier_metrics = metrics.classification_report(label_test.argmax(axis=1), label_pred)

    # save & display metrics
    with open(os.path.join(outpath, 'nn_classifier_metrics.txt'), 'w') as outfile:
        outfile.write(classifier_metrics)

    print(classifier_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "train neural network on the full MNIST dataset and view the classifier metrics")
    parser.add_argument("-d", "--data_path", default = Path('./data/'), type = Path, help = "path to where the MNIST csv-files dataset is saved or where to save it")
    parser.add_argument("-e", "--epochs", default = 5, type = int, help = "numbers of epochs to train")

    args = parser.parse_args()
    
    main(data_path = args.data_path, epochs = args.epochs)
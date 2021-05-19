import os
import argparse
from pathlib import Path

from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils.load_mnist import load_mnist # own util

def main(data_path):
    outpath = 'output'

    # load data as numpy arrays
    img, label = load_mnist(data_path)

    # we are assuming the min and max values for pixel intensities are between 0 and 255
    img = img / 255.0 # normalize pixel vals to between 0 and 1 as float

    # split our data 80/20 - train/test
    img_train, img_test, label_train, label_test = train_test_split(img, label, random_state=1337, test_size=0.2)

    # define the LR model
    classifier = LogisticRegression(penalty='none', 
                                    tol=0.1, 
                                    solver='saga',
                                    multi_class='multinomial').fit(img_train, label_train)

    # do actual prediction on all images in img_test
    label_pred = classifier.predict(img_test)

    # generate comparative metrics with test data
    classifier_metrics = metrics.classification_report(label_test, label_pred)

    # save & display metrics
    with open(os.path.join(outpath, 'lr_classifier_metrics.txt'), 'w') as outfile:
        outfile.write(classifier_metrics)

    print(classifier_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "train logistic regression model on the full MNIST dataset and view the classifier metrics")
    parser.add_argument("-d", "--data_path", default=Path('./data/'), type = Path, help = "path to where the MNIST csv-files dataset is saved or where to save it")
    args = parser.parse_args()
    
    main(data_path = args.data_path)
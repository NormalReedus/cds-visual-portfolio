import os

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

# downloads MNIST data if it does not exist in data_path
def load_mnist(data_path):
    img_path = os.path.join(data_path, 'mnist_img.csv')
    label_path = os.path.join(data_path, 'mnist_label.csv')

    if os.path.isfile(img_path) and os.path.isfile(label_path):
        img = pd.read_csv(img_path)
        label = pd.read_csv(label_path).squeeze() # squeezes DataFrame into a Series
    else:
        if not os.path.isdir(data_path):
            os.mkdir(data_path)

        img, label = fetch_openml('mnist_784', version=1, return_X_y=True)
        img.to_csv(img_path, sep=',', encoding='utf-8', index=False)
        label.to_csv(label_path, sep=',', encoding='utf-8', index=False)

    # we might need to excplicitly convert to numpy arrays for some versions of pandas and sklearn
    return (np.array(img), np.array(label))
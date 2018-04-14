from utils.utils import load_images
import plac
import os
import pdb
import glob
import numpy as np
import random
from models.cnn_models import CNNModel
from keras.preprocessing import image
from sklearn.preprocessing import MultiLabelBinarizer
from keras.applications import imagenet_utils
import tensorflow as tf
random.seed(42)
tf.set_random_seed(42)
np.random.seed(42)

def get_class(file_name):
    c = os.path.basename(file_name).split('.')[0]
    return c

def one_hot_encode(classes):
    class_to_id = { c: idx for idx, c in enumerate(list(set(classes))) }
    y = np.zeros((len(classes), len(class_to_id)))
    for index, c in enumerate(classes):
        y[index, class_to_id[c]] = 1.
    return y

def load_data(directory, train_size, test_size):
    image_paths = glob.glob("{}/*".format(directory))
    random.shuffle(image_paths)

    # Take subset of image paths
    image_paths = image_paths[:train_size + test_size]
    images = load_images(image_paths, (128, 128))

    classes = one_hot_encode([get_class(image_path) for image_path in image_paths])

    X_train = images[:train_size]
    X_test = images[train_size:]
    y_train = classes[:train_size]
    y_test = classes[train_size:]
    return X_train, X_test, y_train, y_test

@plac.annotations(
    cat_or_dog_directory=("Cat or dog directory", "option", "d", str),
    train_size=("Size of training samples", "option", "train_size", int),
    test_size=("Size of test samples", "option", "test_size", int),
)
def main(
        cat_or_dog_directory,
        train_size,
        test_size,
    ):
    X_train, X_test, y_train, y_test = load_data(
        cat_or_dog_directory,
        train_size,
        test_size,
    )
    model = CNNModel('directory')

    for i in range(20):
        model.representation_learning(X_test, epochs=1)
        model.train(X_train, y_train, validation_split=0.1, epochs=1)
    results = model.evaluate(X_test, y_test)
    print(results)

# python evaluate_cats_and_dogs.py -d ../dataset/cat_or_dog/ -train_size 50 -test_size 1000
if __name__ == "__main__":
    plac.call(main)

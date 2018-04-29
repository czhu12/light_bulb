import plac
import os
import pdb
import glob
import numpy as np
import random
from models.rnn_model import RNNModel
from sklearn.model_selection import train_test_split
import tensorflow as tf
random.seed(42)
tf.set_random_seed(42)
np.random.seed(42)

def get_content(file_name):
    return open(file_name).read()

def get_class(file_name):
    return file_name.split('/')[-2]

def one_hot_encode(classes):
    class_to_id = { c: idx for idx, c in enumerate(list(set(classes))) }
    y = np.zeros((len(classes), len(class_to_id)))
    for index, c in enumerate(classes):
        y[index, class_to_id[c]] = 1.
    return y

def load_data(imdb_tagged_directory, train_size, test_size):
    files = glob.glob("{}/**/**.txt".format(imdb_tagged_directory))
    random.shuffle(files)
    files = files[:train_size + test_size]
    content = [get_content(file) for file in files]
    classes = one_hot_encode(np.array([get_class(file) for file in files]))
    X_train = content[:train_size]
    X_test = content[train_size:]
    y_train = classes[:train_size]
    y_test = classes[train_size:]
    
    return X_train, X_test, y_train, y_test

@plac.annotations(
    imdb_tagged_directory=("IMDB reviews tagged directory.", "option", "d", str),
    train_size=("Size of training samples.", "option", "train_size", int),
    test_size=("Size of test samples", "option", "test_size", int),
    epochs=("Num epochs.", "option", "e", int),
)
def main(
        imdb_tagged_directory='/Users/chriszhu/Documents/Github/labelling-tool/dataset/imdb_tagged/',
        train_size=400,
        test_size=1000,
        epochs = 10,
    ):
    X_train, X_test, y_train, y_test = load_data(
        imdb_tagged_directory,
        train_size,
        test_size,
    )
    model = RNNModel('directory')

    model.train(X_train, y_train, validation_split=0.1, epochs=epochs)

    results = model.evaluate(X_test, y_test)
    print(results)

# python evaluate_cats_and_dogs.py -d ../dataset/cat_or_dog/ -train_size 50 -test_size 1000
if __name__ == "__main__":
    plac.call(main)

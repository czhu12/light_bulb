import glob
import os
import logging
import tensorflow as tf

root = logging.getLogger()
root.setLevel(logging.DEBUG)

class BaseModel():
    def __init__(self):
        self.graph = tf.get_default_graph()

    def load_existing_model(self, path):
        self.model.load_weights(path)

    def save(self, path):
        with self.graph.as_default():
            self.model.save(path)

    def train(self, x_train, y_train, validation_split=0., epochs=1):
        raise NotImplementedError()

    def representation_learning(self, x_train, epochs=1):
        raise NotImplementedError()

    def score(self, x):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def evaluate(self, x_test, y_test):
        raise NotImplementedError()




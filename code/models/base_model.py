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

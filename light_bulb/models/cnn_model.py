from utils import utils
import numpy as np
import keras
from models.base_model import BaseModel
from keras_squeezenet import SqueezeNet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Reshape, Conv2D, UpSampling2D, Input
from keras import optimizers
from keras.callbacks import EarlyStopping
from utils import utils

class CNNModel(BaseModel):
    def __init__(
        self,
        num_classes,
        input_shape=(128, 128),
        hyperparameters={
            'loss': 'binary_crossentropy',
            'optimizer': 'SGD',
            'lr': 0.01,
            'embedding_dim': (4, 4, 8),
        },
    ):
        super(CNNModel, self).__init__()

    def representation_learning(self, x_train, epochs=1):
        with self.graph.as_default():
            self.data_generator.fit(x_train)
            results = self.autoencoder.fit(x_train, x_train, epochs=epochs, verbose=0)
            return results.history['loss'][0]

    def fit(self, x_train, y_train, validation_split=0., epochs=1, verbose=0):
        y_train = utils.one_hot_encode(y_train, self.num_classes)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=validation_split,
        )

        if validation_split > 0.:
            validation_data = self.data_generator.flow(x_val, y_val, batch_size=32)
            callbacks = [EarlyStopping(patience=3)]
        else:
            validation_data = None
            callbacks = None

        with self.graph.as_default():
            self.data_generator.fit(x_train)
            results = self.model.fit_generator(
                self.data_generator.flow(x_train, y_train, batch_size=32),
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose,
            )

            return (results.history['loss'][0], results.history['acc'][0])

    def score(self, x):
        with self.graph.as_default():
            return self.model.predict(x)

    def predict(self, x):
        with self.graph.as_default():
            return self.model.predict(x)[:, 1] > 0.5

    def evaluate(self, x_test, y_test, verbose=0):
        y_test = utils.one_hot_encode(y_test, self.num_classes)
        with self.graph.as_default():
            results = self.model.evaluate(x_test, y_test, verbose=verbose)
            return results[0], results[1]

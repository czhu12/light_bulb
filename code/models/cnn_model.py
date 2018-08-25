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
from sklearn.model_selection import train_test_split
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
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model, self.autoencoder = self.initialize_model(input_shape, hyperparameters)
        self.data_generator = self.initialize_data_generator()

        self.initial_weights = self.model.get_weights()

    def reinitialize_model(self):
        self.model.set_weights(self.initial_weights)

    def initialize_data_generator(self):
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        return datagen

    def _encoder_model(self, input_shape, hyperparameters):
        squeezenet = SqueezeNet(
            input_shape=(self.input_shape[0], self.input_shape[1], 3),
            include_top=False,
        )
        x = Flatten()(squeezenet.output)
        embedding = Dense(np.prod(hyperparameters['embedding_dim']), activation='relu')(x)

        encoder = Model(squeezenet.input, embedding)
        utils.freeze_layers(squeezenet)
        return encoder

    def _decoder_model(self, hyperparameters):
        input = Input(shape=(np.prod(hyperparameters['embedding_dim']),))
        x = Reshape(hyperparameters['embedding_dim'])(input)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        decoder = Model(input, decoded)
        decoder.summary()
        return decoder

    def _classifier_model(self, hyperparameters):
        input = Input(shape=(np.prod(hyperparameters['embedding_dim']),))
        x = BatchNormalization()(input)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dropout(0.6)(x)

        x = BatchNormalization()(x)
        x = Dense(self.num_classes)(x)
        x = Activation('softmax')(x)

        classifier = Model(input, x)
        classifier.summary()
        return classifier

    def initialize_model(self, input_shape, hyperparameters):
        encoder = self._encoder_model(input_shape, hyperparameters)
        decoder = self._decoder_model(hyperparameters)
        classifier = self._classifier_model(hyperparameters)

        autoencoder_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)))
        model_input = Input(shape=(self.input_shape[0], self.input_shape[1], 3))
        model = Model(model_input, classifier(encoder(model_input)))

        model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=0.0001),
                metrics=['accuracy'])

        autoencoder.compile(loss='mse',
                optimizer=keras.optimizers.SGD(lr=0.0001),
                metrics=['accuracy'])
        return model, autoencoder

    def representation_learning(self, x_train, epochs=1):
        with self.graph.as_default():
            self.data_generator.fit(x_train)
            results = self.autoencoder.fit(x_train, x_train, epochs=epochs, verbose=0)
            return results.history['loss'][0]

    def fit(self, x_train, y_train, validation_split=0., epochs=1):
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
                verbose=0,
            )

            return (results.history['loss'][0], results.history['acc'][0])

    def score(self, x):
        with self.graph.as_default():
            return self.model.predict(x)

    def predict(self, x):
        with self.graph.as_default():
            return self.model.predict(x)[:, 1] > 0.5

    def evaluate(self, x_test, y_test):
        y_test = utils.one_hot_encode(y_test, self.num_classes)
        with self.graph.as_default():
            results = self.model.evaluate(x_test, y_test, verbose=0)
            return results[0], results[1]

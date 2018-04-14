import pdb
import keras
from utils import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import tensorflow as tf
from utils.glove_utils import LanguageModel
from keras.callbacks import EarlyStopping

class RNNModel():
    def __init__(self, model_directory, config={'word_vectors': False}):
        # maxlen should be computed based on statistic of text lengths.
        self.lang = LanguageModel()
        self.model_directory = model_directory
        self.graph = tf.get_default_graph()
        self.dim_embedding_size = 128
        self.model = self.get_model()

        ## TODO: Download word vectors.
        #if config['word_vectors']:
        #    keras.utils.get_file('http://nlp.stanford.edu/data/glove.6B.zip')

        self.initial_weights = self.model.get_weights()

    def reinitialize_model(self):
        self.model.set_weights(self.initial_weights)

    def representation_learning(self, x_train, epochs=1):
        # Representation learning for text?
        return self

    def get_model(self):
        model = Sequential()
        model.add(self.lang.embedding_layer)
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.summary()

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_texts, y_train, validation_split=0, epochs=1):
        x_train = self.lang.texts_to_sequence(x_texts)
        if validation_split > 0.:
            callbacks = [EarlyStopping(patience=3)]
        else:
            callbacks = None

        with self.graph.as_default():
            return self.model.fit(
                x_train,
                y_train,
                validation_split=validation_split,
                callbacks=callbacks,
                epochs=epochs
            )
    
    def evaluate(self, x_texts, y_test):
        x_test = self.lang.texts_to_sequence(x_texts)
        with self.graph.as_default():
            return self.model.evaluate(x_test, y_test)

    def score(self, x_texts):
        x_test = self.lang.texts_to_sequence(x_texts)
        with self.graph.as_default():
            return self.model.predict(x_test)

    def save(self, name):
        with self.graph.as_default():
            return self.model.save("{}/{}.h5".format(self.model_directory, name))

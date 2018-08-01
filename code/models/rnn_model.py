import pdb
import keras
from models.base_model import BaseModel
from utils import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from utils.glove_utils import LanguageModel
from keras.callbacks import EarlyStopping

class RNNModel(BaseModel):
    def __init__(self, num_classes, config={'word_vectors': False, 'model_type': 'lstm'}):
        # maxlen should be computed based on statistic of text lengths.
        super(RNNModel, self).__init__()
        self.lang = LanguageModel()
        self.dim_embedding_size = 128
        self.num_classes = num_classes
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

    def vectorize_text(self, x_texts):
        x_train, lengths = self.lang.texts_to_sequence(x_texts)
        return x_train

    def get_model(self):
        model = Sequential()
        model.add(self.lang.embedding_layer)
        model.add(Bidirectional(LSTM(100)))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.summary()

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_texts, y_train, validation_split=0, epochs=1):
        x_train = self.vectorize_text(x_texts)
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
                epochs=epochs,
                verbose=0,
            )

    def evaluate(self, x_texts, y_test):
        x_test = self.vectorize_text(x_texts)
        with self.graph.as_default():
            return self.model.evaluate(x_test, y_test)

    def score(self, x_texts):
        x_scores = self.vectorize_text(x_texts)
        with self.graph.as_default():
            return self.model.predict(x_scores)

    def predict(self, x):
        x_scores = self.vectorize_text(x_texts)
        with self.graph.as_default():
            return self.model.predict(x_scores)


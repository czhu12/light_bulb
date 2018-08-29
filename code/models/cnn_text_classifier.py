import time
from models.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from utils.text_utils import WordVectorizer
from utils import utils

class CNNTextClassifier(BaseModel):
    def __init__(
        self,
        num_classes,
        num_filters=250,
        batch_size=32,
        hidden_dims=250,
        kernel_size=3,
    ):
        super(CNNTextClassifier, self).__init__()
        self.num_classes = num_classes
        self.lang = WordVectorizer(use_glove=True)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_dims = hidden_dims
        self.model = self._get_encoder()
        self.model.summary()

    def _get_encoder(self):
        model = Sequential()

        model.add(self.lang.embedding_layer)
        model.add(Dropout(0.2))

        model.add(Conv1D(self.num_filters,
                         self.kernel_size,
                         padding='same',
                         activation='relu',
                         strides=1))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Conv1D(self.num_filters,
                         self.kernel_size,
                         padding='same',
                         activation='relu',
                         strides=1))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(self.hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def vectorize_text(self, x_texts):
        x_train, lengths = self.lang.texts_to_sequence(x_texts)
        return x_train

    def score(self, x_texts):
        start = time.time()
        x_scores = self.vectorize_text(x_texts)
        with self.graph.as_default():
            results = self.model.predict(x_scores)
            print("Inference took: {} seconds".format(time.time() - start))
            return results

    def predict(self, x_texts):
        x_scores = self.vectorize_text(x_texts)
        with self.graph.as_default():
            return self.model.predict(x_scores)

    def representation_learning(self, x_texts):
        pass

    def evaluate(self, x_texts, y_test):
        y_test = utils.one_hot_encode(y_test, self.num_classes)
        x_test = self.vectorize_text(x_texts)
        with self.graph.as_default():
            return self.model.evaluate(x_test, y_test)

    def fit(self, x_texts, y_train, validation_split=0, epochs=1):
        # Freeze language model
        y_train = utils.one_hot_encode(y_train, self.num_classes)
        x_train = self.vectorize_text(x_texts)
        if validation_split > 0.:
            callbacks = [EarlyStopping(patience=3)]
        else:
            callbacks = None

        with self.graph.as_default():
            history = self.model.fit(
                x_train,
                y_train,
                validation_split=validation_split,
                callbacks=callbacks,
                epochs=epochs,
                verbose=0,
            )
            return history.history['loss'][0], history.history['acc'][0]

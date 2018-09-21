from keras.models import Sequential
from keras import layers
from models.base_model import BaseModel
from utils.text_utils import WordVectorizer

class CNNSequenceTagger(BaseModel):
    def __init__(
        self,
        classes,
        num_filters=400,
        kernel_size=3,
        hidden_size=128,
        dropout_rate=0.25,
    ):
        super(CNNSequenceTagger, self).__init__()
        self.lang = WordVectorizer(use_glove=True)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.classes = classes

        self.model = self._get_model()
        self.model.summary()

    def _get_model(self):
        model = Sequential()
        model.add(self.lang.embedding_layer)
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Conv1D(
            self.num_filters,
            self.kernel_size,
            padding='same',
            activation='relu',
            strides=1,
        ))
        
        model.add(layers.Conv1D(
            self.num_filters,
            self.kernel_size,
            padding='same',
            activation='relu',
            strides=1,
        ))
        model.add(layers.TimeDistributed(layers.Dense(self.hidden_size)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(self.dropout_rate))

        model.add(layers.TimeDistributed(layers.Dense(len(self.classes) + 2)))
        model.add(layers.Activation('softmax'))
        model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        return model

    def fit(self, x_seq, y_seq, validation_split=0., epochs=1):
        with self.graph.as_default():
            # Assumption: sequence_tagger(words) -> characters
            x_train, lengths = self.lang.texts_to_sequence(x_seq)
            y_train = self.lang.one_hot_encode_sequence(y_seq, self.classes)
            import pdb
            pdb.set_trace()
            return self.model.fit(x_train, y_train)

    def representation_learning(self, x_train, epochs=1):
        with self.graph.as_default():
            pass

    def score(self, x_seq):
        with self.graph.as_default():
            # Unroll until completion (this returns a string)
            x_seq, lengths = self.lang.texts_to_sequence(x_seq)
            return self.model.predict(x_seq)

    def predict(self, x):
        with self.graph.as_default():
            x, lengths = self.lang.texts_to_sequence(x)
            return self.lang.decode_one_hot_sequence_predictions(
                self.model.predict(x),
                lengths,
                self.classes,
            )

    def evaluate(self, x_seq, y_seq):
        with self.graph.as_default():
            x_eval, lengths = self.lang.texts_to_sequence(x_seq)
            y_eval = self.lang.one_hot_encode_sequence(y_seq, self.classes)
            return self.model.evaluate(x_eval, y_eval)

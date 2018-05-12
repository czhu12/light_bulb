import keras
from keras import layers
from keras.models import Sequential
from nltk import word_tokenize
import string
from utils.glove_utils import LanguageModel
import random
import numpy as np

from models.base_model import BaseModel

# Character seq2seq model. This way we don't need to worry about vocab size
# This class has a few capabilities:
# input_type are (1) words (glove) or (2) characters
# Outputs are either: (1) characters or (2) valid_outputs
# Model is either a (1) seq2seq model or (2) many to many sequence tagger

HIDDEN_SIZE = 128

class SequenceModel(BaseModel):
    def __init__(
            self,
            valid_outputs=None,
            seq2seq=False,
            character_mode=False,
        ):
        super(SequenceModel, self).__init__()
        self.seq2seq = seq2seq
        self.character_mode = character_mode
        self.valid_outputs = valid_outputs

        self.lang = LanguageModel()
        self.model = self.get_model()
        # seq2seq or TimeDistributed(RNN)

    def get_model(self):
        with self.graph.as_default():
            if self.seq2seq:
                raise NotImplementedError('Seq2seq model not yet implemented')
            model = Sequential()
            model.add(self.lang.embedding_layer)
            model.add(layers.Bidirectional(layers.LSTM(HIDDEN_SIZE, return_sequences=True)))
            # We need + 2 for the EOS_TOKEN and the PAD_TOKEN. We should come up with a better way to do this
            model.add(layers.TimeDistributed(layers.Dense(len(self.valid_outputs) + 2, activation='softmax')))
            model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])
            return model

    def train(self, x_seq, y_seq, validation_split=0., epochs=1):
        with self.graph.as_default():
            # Assumption: sequence_tagger(words) -> characters
            x_train, lengths = self.lang.texts_to_sequence(x_seq)
            y_train = self.lang.one_hot_encode_sequence(y_seq, valid_tokens=self.valid_outputs)
            return self.model.fit(x_train, y_train)

    def representation_learning(self, x_train, epochs=1):
        with self.graph.as_default():
            pass

    def score(self, x_seq):
        with self.graph.as_default():
            if self.seq2seq:
                raise ValueError("Can't do sequence to sequence models yet.")
            else:
                # Unroll until completion (this returns a string)
                x_seq, lengths = self.lang.texts_to_sequence(x_seq, character=self.character_mode)
                return self.model.predict(x_seq)

    def predict(self, x):
        with self.graph.as_default():
            x, lengths = self.lang.texts_to_sequence(x)
            return self.lang.decode_one_hot_sequence_predictions(
                self.model.predict(x),
                lengths,
                valid_tokens=self.valid_outputs,
            )

    def evaluate(self, x_seq, y_seq):
        with self.graph.as_default():
            x_eval, lengths = self.lang.texts_to_sequence(x_seq)
            y_eval = self.lang.one_hot_encode_sequence(y_seq, valid_tokens=self.valid_outputs)
            return self.model.evaluate(x_eval, y_eval)

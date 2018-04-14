import pdb
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from nltk.tokenize import word_tokenize

EMBEDDING_DIM = 50
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
UNKNOWN_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

class LanguageModel():
    def __init__(self, path='/Users/chriszhu/Downloads/glove.6B/glove.6B.50d.txt'):
        embedding_matrix = np.zeros(())
        embeddings_index = {}
        f = open(path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        num_words = len(embeddings_index) + 3
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        word2index = { UNKNOWN_TOKEN: 0, EOS_TOKEN: 1, PAD_TOKEN: 2 }

        for index, (word, embedding) in enumerate(embeddings_index.items()):
            embedding_matrix[index + 3] = embedding
            word2index[word] = index + 3

        self.embedding_layer = Embedding(
                num_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                trainable=False)
        self.vocab = list(embeddings_index.keys())
        self.num_words = num_words
        self.embedding_matrix = embedding_matrix
        self.word2index = word2index

    def _embedding(self, word, word2index):
        if word in word2index:
            return word2index[word]
        return word2index[UNKNOWN_TOKEN]

    def _sequence_ids(self, texts):
        sequences = []
        for text in texts:
            words = word_tokenize(text.lower()) + [EOS_TOKEN]
            ids = [self._embedding(word, self.word2index) for word in words]
            sequences.append(ids)
        return sequences

    def text_embedding(self, texts):
        sequences = self._sequence_ids(texts)
        text_embeddings = []
        for ids in sequences:
            word_embeddings = self.embedding_matrix[ids]
            text_embedding = word_embeddings.mean(1)
            text_embeddings.append(text_embedding)
        assert len(text_embeddings) == len(texts)
        return text_embeddings
        
    def texts_to_sequence(self, texts, maxlen=None):
        """
        Sequences is padded of size (batch, maxlen).
        """
        sequences = self._sequence_ids(texts)

        if not maxlen:
            maxlen = max([len(ids) for ids in sequences])

        return pad_sequences(sequences, maxlen=maxlen)

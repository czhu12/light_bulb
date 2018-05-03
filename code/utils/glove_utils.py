import pdb
import os
import sys
import string
import logging
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from nltk.tokenize import word_tokenize
import requests
from subprocess import call

EMBEDDING_DIM = 50
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
UNKNOWN_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

logger = logging.getLogger()

def download_glove_vectors(remote_path='http://nlp.stanford.edu/data/glove.6B.zip'):
    if os.path.exists('./vendor/glove/glove.6B'):
        logger.debug("Already downloaded glove vectors")
        return './vendor/glove/glove.6B/glove.6B.50d.txt'

    if not os.path.isdir('./vendor'):
        logger.debug("Created ./vendor")
        os.makedirs('./vendor')

    if not os.path.isdir('./vendor/glove'):
        logger.debug("Created ./vendor/glove")
        os.makedirs('./vendor/glove')

    with open('./vendor/glove/glove.6B.zip', "wb") as f:
        response = requests.get(remote_path)
        f.write(response.content)
        call(['unzip', './vendor/glove/glove.6B.zip', '-d', './vendor/glove/glove.6B'])
        logger.debug("Unzipped to './vendor/glove/glove.6B'")

    return './vendor/glove/glove.6B/glove.6B.50d.txt'


class Vocabulary(object):
    def __init__(self, vocabulary):
        self.vocab = [PAD_TOKEN, EOS_TOKEN, UNKNOWN_TOKEN] + vocabulary
        self.num_words = len(self.vocab)
        self.word2index = { v: i for i, v in enumerate(self.vocab) }
        self.embedding_layer = Embedding(
                len(vocab),
                EMBEDDING_DIM,
                trainable=True)

    def _embedding(self, word, word2index):
        if word in word2index:
            return word2index[word]
        return word2index[UNKNOWN_TOKEN]

    def _sequence_ids(self, texts, character=False):
        sequences = []
        for text in texts:
            if character:
                words = list(text) + [EOS_TOKEN]
            else:
                words = word_tokenize(text.lower()) + [EOS_TOKEN]

            ids = [self._embedding(word, self.word2index) for word in words]
            sequences.append(ids)
        return sequences

    def decode_one_hot_sequence_predictions(self, y_scores, valid_tokens=None):
        if not valid_tokens:
            valid_tokens = string.printable

        id2token = [PAD_TOKEN, EOS_TOKEN] + list(valid_tokens)
        token2id = { t: i for i, t in enumerate(id2token) }
        decoded = []
        for index, y_score in enumerate(y_scores):
            tags = []
            idxs = np.argmax(y_score, axis=1)
            for idx in idxs:
                tags.append(id2token[idx])

            decoded.append(tags)
        # Filter out all preset tokens
        return decoded

    # TODO: Need to add words support to this
    def one_hot_encode_sequence(self, y_seqs, valid_tokens=None):
        if valid_tokens:
            y_seqs = [y_seq.split(' ') for y_seq in y_seqs]
        else:
            valid_tokens = string.printable

        valid_tokens = set(valid_tokens)
        id2token = [PAD_TOKEN, EOS_TOKEN] + list(valid_tokens)
        token2id = { t: i for i, t in enumerate(id2token) }
        # Validate y_seqs
        for y_seq in y_seqs:
            assert all(y in valid_tokens for y in y_seq), "{} not in valid_tokens".format(y_seq, valid_tokens)

        def _to_seq_ids(seq):
            return [token2id[y] for y in seq] + [token2id[EOS_TOKEN]]

        y_seq_ids = [_to_seq_ids(y_seq) for y_seq in y_seqs]

        ys = pad_sequences(y_seq_ids)
        y_one_hot = np.zeros((len(ys), len(ys[0]), len(token2id)))

        for idx, y in enumerate(ys):
            for idx2, y2 in enumerate(y):
                y_one_hot[idx][idx2][y2] = 1.

        return y_one_hot

    def texts_to_sequence(self, texts, maxlen=None, character=False):
        """
        Sequences is padded of size (batch, maxlen).
        """
        sequences = self._sequence_ids(texts, character)

        if not maxlen:
            maxlen = max([len(ids) for ids in sequences])

        return pad_sequences(sequences, maxlen=maxlen)

class LanguageModel(Vocabulary):
    def __init__(self):
        path = download_glove_vectors()
        self.load_glove_vectors(path)

    def load_glove_vectors(self, path):
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
        word2index = { PAD_TOKEN: 0, UNKNOWN_TOKEN: 1, EOS_TOKEN: 2 }

        for index, (word, embedding) in enumerate(embeddings_index.items()):
            embedding_matrix[index + 3] = embedding
            word2index[word] = index + 3

        self.embedding_layer = Embedding(
                num_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True)

        self.vocab = list(embeddings_index.keys())
        self.num_words = num_words
        self.embedding_matrix = embedding_matrix
        self.word2index = word2index

    def text_embedding(self, texts):
        sequences = self._sequence_ids(texts)
        embeddings = []
        for ids in sequences:
            word_embeddings = self.embedding_matrix[ids]
            embedding = word_embeddings.mean(1)
            embeddings.append(embedding)
        assert len(embeddings) == len(texts)
        return text_embeddings

if __name__ == "__main__":
    lang = LanguageModel()
    import pdb
    pdb.set_trace()
    lang.one_hot_encode_sequence( [['O'], ['B-NP O'], ['B-NP I-NP']], valid_tokens=['O', 'B-NP', 'I-NP'],)

import pdb
import logging
import numpy as np
#from keras.preprocessing.sequence import pad_sequences
#from nltk.tokenize import word_tokenize
import requests
#from nltk.tokenize.toktok import ToktokTokenizer
#from keras.layers import Embedding

import re
import html
from concurrent.futures import ProcessPoolExecutor



EMBEDDING_DIM = 50
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
UNKNOWN_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

logger = logging.getLogger()

class WordVectorizer():
    def __init__(
        self,
        index2word=[],
        use_glove=False,
        glove_path='vendor/glove.6B/glove.6B.50d.txt',
    ):
        if use_glove:
            index2word, embedding_layer = self._load_glove_vectors(glove_path)
            self.embedding_layer = embedding_layer

        self.index2word = index2word

        self.word2index = { word: index for index, word in enumerate(self.index2word) }
        self.tokenizer = ToktokTokenizer()

    def _load_glove_vectors(self, path):
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

        embedding_layer = Embedding(
                num_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                trainable=False)

        vocab = [PAD_TOKEN, UNKNOWN_TOKEN, EOS_TOKEN] + list(embeddings_index.keys())
        return vocab, embedding_layer


    def _embedding(self, word, word2index):
        if word in word2index:
            return word2index[word.lower()]
        return word2index[UNKNOWN_TOKEN]

    def _tokenize(self, texts, include_stop_token=False):
        tokenized_texts = []
        for text in texts:
            words = self.tokenizer.tokenize(text)

            if include_stop_token:
                words += [EOS_TOKEN]
            tokenized_texts.append(words)

        return tokenized_texts

    def _sequence_ids(self, tokenized, include_stop_token=False):
        sequences = []
        for tokens in tokenized:
            ids = [self._embedding(token, self.word2index) for token in tokens]
            sequences.append(ids)

        return sequences

    def texts_to_sequence(self, texts, maxlen=None, include_stop_token=False):
        """
        Sequences is padded of size (batch, maxlen).
        """
        tokenized = self._tokenize(texts, include_stop_token=include_stop_token)
        return self.tokenized_to_sequence(tokenized, maxlen=maxlen, include_stop_token=include_stop_token)

    def tokenized_to_sequence(self, tokenized, maxlen=None, include_stop_token=False):
        """
        Sequences is padded of size (batch, maxlen).
        """
        sequences = self._sequence_ids(tokenized, include_stop_token=include_stop_token)
        lengths = [len(ids) for ids in sequences]

        if not maxlen:
            maxlen = max([len(ids) for ids in sequences])

        return pad_sequences(sequences, maxlen=maxlen), lengths


    def text_embedding(self, texts):
        sequences = self._sequence_ids(texts)
        embeddings = []
        for ids in sequences:
            word_embeddings = self.embedding_matrix[ids]
            embedding = word_embeddings.mean(1)
            embeddings.append(embedding)
        assert len(embeddings) == len(texts)
        return text_embeddings

def flatten(l):
    return [item for sublist in l for item in sublist]

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'")\
        .replace('nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n")\
        .replace('quot;', "'").replace('<br />', "\n").replace('\\"', '"')\
        .replace('<unk>','u_n').replace(' @.@ ','.').replace(' @-@ ','-').replace('\\', ' \\ ')
    return re.sub(r' +', ' ', html.unescape(x))

if __name__ == "__main__":
    lang = LanguageModel()
    import pdb
    pdb.set_trace()
    lang.one_hot_encode_sequence( [['O'], ['B-NP O'], ['B-NP I-NP']], valid_tokens=['O', 'B-NP', 'I-NP'],)

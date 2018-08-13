import plac
import os
import pdb
import numpy as np
from subprocess import call
from utils.utils import download_file
from keras.layers import Embedding
from models.rnn_model import RNNModel
from nltk.tokenize.toktok import ToktokTokenizer

def download_glove_vectors(remote_path='http://nlp.stanford.edu/data/glove.6B.zip'):
    if os.path.exists('./vendor/glove/glove.6B'):
        logger.debug("Already downloaded glove vectors")
        return './vendor/glove/glove.6B/glove.6B.50d.txt'

    downloaded_filepath = download_file(remote_path, './vendor/glove')
    call(['unzip', downloaded_filepath, '-d', './vendor/glove/glove.6B'])
    logger.debug("Unzipped to './vendor/glove/glove.6B'")

    return './vendor/glove/glove.6B/glove.6B.50d.txt'

class Vocab:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size
    def load_glove_vectors(self, path):
        embeddings_index = {}
        f = open(path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        word2index = { '<pad>': 0, '<unk>': 1, '<eos>': 2 }
        num_words = len(embeddings_index) + len(word2index)
        embedding_matrix = np.zeros((num_words, self.embedding_size))

        for index, (word, embedding) in enumerate(embeddings_index.items()):
            embedding_matrix[len(word2index)] = embedding
            word2index[word] = len(word2index)

        self.embedding_layer = Embedding(
            num_words,
            200,
            weights=[embedding_matrix],
            trainable=False,
            mask_zero=True,
        )

        self.vocab = list(word2index.keys())
        self.num_words = num_words
        self.embedding_matrix = embedding_matrix
        self.word2index = word2index

def create_data(lines, bptt=70):
    tokenizer = ToktokTokenizer()
    lines = [line.lower() for line in lines if len(line) > 40]
    all_text = ' \n '.join(lines)
    tokenized = tokenizer.tokenize(all_text)
    chunks = [tokenized[i:i + bptt + 1] for i in range(0, len(tokenized), bptt + 1)]
    chunks = [' '.join(chunk) for chunk in chunks]

    return chunks

# Train a language model
@plac.annotations(
    wikitext2_path=("Path to wikitext-2 directory.", "option", "d", str),
    save_dir=("Location to save pretrained model", "option", "o", str))
def main(wikitext2_path, save_dir):
    lines = open(os.path.join(wikitext2_path, 'wiki.train.tokens'), 'r').readlines()
    bptt = 150
    text_batches = create_data(lines, bptt=bptt)
    # Modelling part
    embedding_size = 200
    vocab = Vocab(embedding_size)
    vocab.load_glove_vectors('vendor/glove/glove.6B/glove.6B.{}d.txt'.format(embedding_size))
    model = RNNModel(2, embedding_size, vocab.vocab)
    model.representation_learning(text_batches, verbose=True)
    pdb.set_trace()

if __name__ == '__main__':
    plac.call(main)

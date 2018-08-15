import plac
import os
import pdb
import numpy as np
import pickle
from collections import Counter
from subprocess import call
from utils.utils import download_file
from keras.layers import Embedding
from keras.models import load_model
from models.rnn_model import RNNModel
from nltk.tokenize.toktok import ToktokTokenizer

def download_glove_vectors(remote_path='http://nlp.stanford.edu/data/glove.6B.zip'):
    if os.path.exists('./vendor/glove/glove.6B'):
        return './vendor/glove/glove.6B/glove.6B.50d.txt'

    downloaded_filepath = download_file(remote_path, './vendor/glove')
    call(['unzip', downloaded_filepath, '-d', './vendor/glove/glove.6B'])

    return './vendor/glove/glove.6B/glove.6B.50d.txt'

class Vocab:
    def __init__(self, wikitext_path, max_vocab_size=100000):
        self.wikitext_path = wikitext_path
        self.tokenizer = ToktokTokenizer()
        test_counts = self.process_tokens(os.path.join(wikitext_path, 'wiki.test.tokens'))
        train_counts = self.process_tokens(os.path.join(wikitext_path, 'wiki.train.tokens'))
        valid_counts = self.process_tokens(os.path.join(wikitext_path, 'wiki.valid.tokens'))
        counts = test_counts + train_counts + valid_counts
        self.vocab = [word for word, count in counts.most_common(max_vocab_size) if count > 1]
        self.vocab = ['<pad>', '<eos>'] + self.vocab

    def process_tokens(self, path):
        return Counter(self.tokenizer.tokenize(open(path, 'r').read()))

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
    save_dir=("Location to save pretrained model", "option", "o", str),
    num_gpus=("Number of GPU's to use", "option", "n", int),
    mode=("Mode [`eval` or `train`]", "option", "m", str))
def main(wikitext2_path, save_dir, num_gpus, mode='train'):
    lines = open(os.path.join(wikitext2_path, 'wiki.train.tokens'), 'r').readlines()
    bptt = 150
    text_batches = create_data(lines, bptt=bptt)
    # Modelling part
    embedding_size = 200
    vocab_path = os.path.join(save_dir, 'vocab.p')
    if os.path.exists(save_dir):
        model = RNNModel(2, embedding_size, vocab)
        model.load(save_dir)
        vocab = pickle.load(vocab_path)
    else:
        vocab = Vocab(wikitext2_path).vocab
        model = RNNModel(2, embedding_size, vocab)

    if mode == 'train':
        model.representation_learning(
            text_batches,
            verbose=True,
            epochs=30,
            num_gpus=num_gpus,
        )
        os.makedirs(save_dir)
        model.save(save_dir)
        pickle.dump(vocab, open(vocab_path, 'wb'))
    else:
        model.representation_learning(text_batches, evaluate=True)
        print("Evaluation is not implemented yet.")

if __name__ == '__main__':
    plac.call(main)

import pdb
import sys
import string
import logging
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import requests
from nltk.tokenize.toktok import ToktokTokenizer

import re
import spacy
import html
from spacy.symbols import ORTH
from concurrent.futures import ProcessPoolExecutor


EMBEDDING_DIM = 50
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
UNKNOWN_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

logger = logging.getLogger()

class WordVectorizer():
    def __init__(self, index2word):
        self.index2word = index2word
        self.word2index = { word: index for index, word in enumerate(index2word) }
        self.tokenizer = ToktokTokenizer()

    def _embedding(self, word, word2index):
        if word in word2index:
            return word2index[word]
        return word2index[UNKNOWN_TOKEN]

    def _tokenize(self, texts, character=False, include_stop_token=False):
        tokenized_texts = []
        for text in texts:
            if character:
                words = list(text)
            else:
                words = self.tokenizer.tokenize(text.lower())

            if include_stop_token:
                words += [EOS_TOKEN]
            tokenized_texts.append(words)

        return tokenized_texts

    def _sequence_ids(self, tokenized, character=False, include_stop_token=False):
        sequences = []
        for tokens in tokenized:
            ids = [self._embedding(token, self.word2index) for token in tokens]
            sequences.append(ids)

        return sequences

    def decode_one_hot_sequence_predictions(self, y_scores, lengths, valid_tokens=None):
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

            length = lengths[index]
            decoded.append(tags[-length:-1])
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

    def texts_to_sequence(self, texts, maxlen=None, character=False, include_stop_token=False):
        """
        Sequences is padded of size (batch, maxlen).
        """
        tokenized = self._tokenize(texts, character, include_stop_token=include_stop_token)
        sequences = self._sequence_ids(tokenized, character, include_stop_token=include_stop_token)
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

class Tokenizer():
    re_rep      = re.compile(r'(\S)(\1{3,})')
    re_word_rep = re.compile(r'(\b\w+\W+)(\1{3,})')
    re_br       = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)

    def __init__(self, lang='en'):
        self.tok = spacy.load(lang)
        for w in ('<eos>','<bos>','<unk>'):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def spacy_tok(self,x):
        return [t.text for t in self.tok.tokenizer(self.re_br.sub("\n", x))]

    @staticmethod
    def replace_rep(m):
        TK_REP = 'tk_rep'
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '

    @staticmethod
    def replace_wrep(m):
        TK_WREP = 'tk_wrep'
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '

    @staticmethod
    def do_caps(ss):
        TOK_UP = ' t_up '
        res = [[TOK_UP, s.lower()] if (s.isupper() and (len(s) > 2)) else [s.lower()] for s in re.findall(r'\w+|\W+', ss)]
        return ''.join(sum(res, []))

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r'([/#])', r' \1 ', s)
        s = re.sub(' {2,}', ' ', s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang='en'):
        tok = Tokenizer(lang)
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss, lang='en', ncpus=32):
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang] * len(ss)), [])

if __name__ == "__main__":
    lang = LanguageModel()
    import pdb
    pdb.set_trace()
    lang.one_hot_encode_sequence( [['O'], ['B-NP O'], ['B-NP I-NP']], valid_tokens=['O', 'B-NP', 'I-NP'],)

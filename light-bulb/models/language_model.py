#!/usr/bin/env python

"""
    ulmfit.py

    ULMFIT functions

    !! Most of these functions are copied exactly/approximately from the `fastai` library.
        https://github.com/fastai/fastai

    !! This will not work on torch>=0.4, due to torch bugs
"""

import re
import sys
import json
import warnings
import numpy as np
from time import time
import pickle
import numpy as np
import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from utils.tokenizer import Tokenizer
from torch.utils.data import DataLoader

import pdb

# --
# Helpers
def flatten(l):
    return [item for sublist in l for item in sublist]

class TextVectorizer:
    @staticmethod
    def _vectorize(tokenized_text, stoi, unknown_idx=0):
        return np.array([stoi[token] if token in stoi else unknown_idx for token in tokenized_text])

    @staticmethod
    def vectorize(tokenized_texts, stoi, unknown_idx=0):
        return [TextVectorizer._vectorize(
            tokenized_text,
            stoi,
            unknown_idx
        ) for tokenized_text in tokenized_texts]

class RaggedDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), 'len(X) != len(y)'
        self.X = [torch.LongTensor(xx) for xx in X]
        self.y = torch.LongTensor(y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class SortishSampler(Sampler):
    # adapted from `fastai`
    def __init__(self, data_source, batch_size, batches_per_chunk=50):
        self.data_source       = data_source
        self._key              = lambda idx: len(data_source[idx])
        self.batch_size        = batch_size
        self.batches_per_chunk = batches_per_chunk

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):

        idxs = np.random.permutation(len(self.data_source))

        # Group records into batches of similar size
        chunk_size = self.batch_size * self.batches_per_chunk
        chunks     = [idxs[i:i+chunk_size] for i in range(0, len(idxs), chunk_size)]
        idxs       = np.hstack([sorted(chunk, key=self._key, reverse=True) for chunk in chunks])

        # Make sure largest batch is in front (for memory management reasons)
        batches         = [idxs[i:i+self.batch_size] for i in range(0, len(idxs), self.batch_size)]
        batch_order     = np.argsort([self._key(b[0]) for b in batches])[::-1]
        batch_order[1:] = np.random.permutation(batch_order[1:])

        idxs = np.hstack([batches[i] for i in batch_order])
        return iter(idxs)


def text_collate_fn(batch, pad_value=1):
    X, y = zip(*batch)

    max_len = max([len(xx) for xx in X])
    X = [F.pad(xx, pad=(max_len - len(xx), 0), value=pad_value).data for xx in X]

    X = torch.stack(X, dim=-1)
    y = torch.LongTensor(y)
    return X, y


def to_numpy(x):
    if type(x) in [np.ndarray, float, int]:
        return x
    elif isinstance(x, Variable):
        return to_numpy(x.data)
    else:
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()

def get_children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_freeze(x, mode):
    x.frozen = mode
    for p in x.parameters():
        p.requires_grad = not mode

    for module in get_children(x):
        set_freeze(module, mode)


def load_lm_weights(lm_weights_path, lm_itos_path, itos_path):
    lm_weights = torch.load(lm_weights_path, map_location=lambda storage, loc: storage)

    lm_itos = pickle.load(open(lm_itos_path, 'rb'))
    lm_stoi = {v:k for k,v in enumerate(lm_itos)}

    itos = pickle.load(open(itos_path, 'rb'))
    n_tok = len(itos)

    # Adjust vocabulary to match finetuning corpus
    lm_enc_weights = to_numpy(lm_weights['0.encoder.weight'])

    pdb.set_trace()
    tmp = np.zeros((n_tok, lm_enc_weights.shape[1]), dtype=np.float32)
    tmp += lm_enc_weights.mean(axis=0)
    for i, w in enumerate(itos):
        if w in lm_stoi:
            tmp[i] = lm_enc_weights[lm_stoi[w]]

    lm_weights['0.encoder.weight']                    = torch.Tensor(tmp.copy())
    lm_weights['0.encoder_with_dropout.embed.weight'] = torch.Tensor(tmp.copy())
    lm_weights['1.decoder.weight']                    = torch.Tensor(tmp.copy())

    return lm_weights, n_tok


def detach(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return tuple([detach(xx) for xx in x])
    # elif IS_TORCH_04:
    #     return x.detach()
    else:
        return Variable(x.data)

# --
# LM dataloader

class LanguageModelLoader():
    # From `fastai`
    def __init__(self, data, bs, bptt, backwards=False):
        self.bs        = bs
        self.bptt      = bptt
        self.backwards = backwards
        self.data      = self.batchify(data, bs)
        self.i         = 0
        self.iter      = 0
        self.n         = len(self.data)

    def batchify(self, data, bs):
        trunc = data.shape[0] - data.shape[0] % bs
        data = np.array(data[:trunc])

        data = data.reshape(bs, -1).T

        if self.backwards:
            data = data[::-1]

        return torch.LongTensor(np.ascontiguousarray(data))

    def __iter__(self):
        self.i    = 0
        self.iter = 0
        while (self.i < self.n - 1) and (self.iter < len(self)):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))

            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def get_batch(self, i, seq_len):
        seq_len = min(seq_len, self.data.shape[0] - 1 - i)
        return self.data[i:(i+seq_len)], self.data[(i+1):(i+seq_len+1)].view(-1)

    def __len__(self):
        return self.n // self.bptt - 1

# --
# RNN Encoder

def dropout_mask(x, sz, dropout):
    # From `fastai`
    return x.new(*sz).bernoulli_(1 - dropout)/ (1 - dropout)


class LockedDropout(nn.Module):
    # From `fastai` and `salesforce/awd-lstm-lm`
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p:
            return x
        else:
            mask = dropout_mask(x.data, (1, x.shape[1], x.shape[2]), self.p)
            mask = Variable(mask, requires_grad=False)
            return mask * x

    def __repr__(self):
        return 'LockedDropout(p=%f)' % self.p


class WeightDrop(torch.nn.Module):
    # From `fastai` and `salesforce/awd-lstm-lm`
    def __init__(self, module, dropout, weights=['weight_hh_l0']):
        super().__init__()
        self.module  = module
        self.weights = weights
        self.dropout = dropout

        if isinstance(self.module, torch.nn.RNNBase):
            def noop(*args, **kwargs): return
            self.module.flatten_parameters = noop

        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

    def __repr__(self):
        return 'WeightDrop(%s)' % self.module.__repr__()


class EmbeddingDropout(nn.Module):
    # From `fastai` and `salesforce/awd-lstm-lm`
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, words, dropout=0.1, scale=None):
        if dropout:
            mask = dropout_mask(self.embed.weight.data, (self.embed.weight.size(0), 1), dropout)
            mask = Variable(mask)
            masked_embed_weight = mask * self.embed.weight
        else:
            masked_embed_weight = self.embed.weight

        if scale:
            masked_embed_weight = scale * masked_embed_weight

        padding_idx = self.embed.padding_idx
        if padding_idx is None:
            padding_idx = -1

        # if IS_TORCH_04:
        #     X = F.embedding(words,
        #         masked_embed_weight, padding_idx, self.embed.max_norm,
        #         self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse)
        # else:
        return self.embed._backend.Embedding.apply(words,
            masked_embed_weight, padding_idx, self.embed.max_norm,
            self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse)

    def __repr__(self):
        return 'EmbeddingDropout(%s)' % self.embed.__repr__()


class RNN_Encoder(nn.Module):
    # From `fastai`
    def __init__(self, n_tok, emb_sz, nhid, nlayers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5, initrange=0.1):

        super().__init__()

        self.emb_sz     = emb_sz
        self.nhid       = nhid
        self.nlayers    = nlayers
        self.dropoute   = dropoute
        self.ndir       = 2 if bidir else 1
        self.batch_size = 1

        self.encoder = nn.Embedding(n_tok, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        self.dropouti = LockedDropout(dropouti)

        self.rnns = [
            nn.LSTM(
                input_size=emb_sz if l == 0 else nhid,
                hidden_size=(nhid if l != nlayers - 1 else emb_sz) // self.ndir,
                num_layers=1,
                bidirectional=bidir,
                dropout=dropouth
            ) for l in range(nlayers)
        ]
        self.rnns = [WeightDrop(rnn, dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(nlayers)])

        self.encoder.weight.data.uniform_(-initrange, initrange)

    def one_hidden(self, l):
        nh = (self.nhid if l != self.nlayers - 1 else self.emb_sz) // self.ndir
        return Variable(self.weights.new(self.ndir, self.batch_size, nh).zero_(), volatile=not self.training)

    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.nlayers)]

    def forward(self, x):
        batch_size = x.shape[1]
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.reset()

        emb = self.encoder_with_dropout(x, dropout=self.dropoute if self.training else 0)
        emb = self.dropouti(emb)

        raw_output = emb
        new_hidden, raw_outputs, outputs = [], [], []
        for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_output, new_h = rnn(raw_output, self.hidden[l])

            new_hidden.append(new_h)
            raw_outputs.append(raw_output)

            if l != self.nlayers - 1:
                raw_output = drop(raw_output)

            outputs.append(raw_output)

        self.hidden = detach(new_hidden)
        return raw_outputs, outputs


# --
# LM classes

class LinearDecoder(nn.Module):
    # From `fastai`
    def __init__(self, in_features, out_features, dropout, decoder_weights=None, initrange=0.1):
        super().__init__()

        self.decoder = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if decoder_weights:
            self.decoder.weight = decoder_weights.weight

        self.dropout = LockedDropout(dropout)

    def forward(self, input):
        _, x = input
        x = self.dropout(x[-1])
        x = x.view(x.size(0) * x.size(1), x.size(2))
        x = self.decoder(x)
        x = x.view(-1, x.size(1))

        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        itos,
        emb_sz=400,
        nhid=1150,
        nlayers=3,
        pad_token=1,
        dropout_scale=0.7,
        bptt=70,
        max_seq=20 * 70,
        tie_weights=True,
    ):
        super(LanguageModel, self).__init__()
        drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * dropout_scale
        dropouti  = drops[0]
        dropout   = drops[1]
        wdrop     = drops[2]
        dropoute  = drops[3]
        dropouth  = drops[4]
        self.itos = itos
        self.stoi = { s: i for i, s in enumerate(itos) }

        self.emb_sz = emb_sz
        self.encoder = MultiBatchRNN(
            bptt=bptt,
            max_seq=max_seq,
            n_tok=len(itos),
            emb_sz=emb_sz,
            nhid=nhid,
            nlayers=nlayers,
            pad_token=pad_token,
            dropouth=dropouth,
            dropouti=dropouti,
            dropoute=dropoute,
            wdrop=wdrop,
        )
        self.encoder.reset()

        self.decoder = LinearDecoder(
            in_features=emb_sz,
            out_features=len(itos),
            dropout=dropout,
            decoder_weights=self.encoder.encoder if tie_weights else None
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_layer_groups(self):
        return [
            *zip(self.encoder.rnns, self.encoder.dropouths),
            (self.decoder, self.encoder.dropouti)
        ]

    def reset(self):
        _ = [c.reset() for c in self.children() if hasattr(c, 'reset')]

    def load_weights(self, wgts):
        tmp = {}
        for k,v in wgts.items():
            k = re.sub(r'^0.', 'encoder.', k)
            k = re.sub(r'^1.', 'decoder.', k)
            tmp[k] = v

        self.load_state_dict(tmp)

# --
# Classifier classes


class MultiBatchRNN(RNN_Encoder):
    # From `fastai`
    def __init__(self, bptt, max_seq, predict_only=False, *args, **kwargs):
        self.max_seq      = max_seq
        self.bptt         = bptt
        self.predict_only = predict_only
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, x):
        sl  = x.shape[0]
        _ = [[hh.data.zero_() for hh in h] for h in self.hidden]

        raw_outputs, outputs = [], []
        for i in range(0, sl, self.bptt):
            raw_output, output = super().forward(x[i: min(i + self.bptt, sl)])
            if i > (sl - self.max_seq):
                raw_outputs.append(raw_output)
                outputs.append(output)

        return self.concat(raw_outputs), self.concat(outputs)


class PoolingLinearClassifier(nn.Module):
    # Adapted from `fastai`
    def __init__(self, layers, drops, predict_only=False):
        super().__init__()

        self.predict_only = predict_only

        self.layers = []
        for i in range(len(layers) - 1):
            self.layers += [
                nn.BatchNorm1d(num_features=layers[i]),
                nn.Dropout(p=drops[i]),
                nn.Linear(in_features=layers[i], out_features=layers[i + 1]),
                nn.ReLU(),
            ]

        self.layers.pop() # Remove last relu

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        raw_outputs, outputs = x
        last_raw_output, last_output = raw_outputs[-1], outputs[-1]

        x = torch.cat([
            last_output[-1],
            last_output.max(dim=0)[0],
            last_output.mean(dim=0)
        ], 1)

        if self.predict_only:
            return self.layers(x)
        else:
            return self.layers(x), last_raw_output, last_output


class LM_TextClassifier(nn.Module):
    def __init__(
        self,
        language_model,
        n_classes,
        predict_only=False,
    ):
        super(LM_TextClassifier, self).__init__()

        self.encoder = language_model
        self.decoder = PoolingLinearClassifier(
            [self.encoder.emb_sz * 3, 50, n_classes],
            [0.2, 0.2],
            predict_only=predict_only,
        )

    def representation_learning(self, x_texts):
        tokenized = np.array(flatten(Tokenizer().proc_all(x_texts)))
        vectorized = TextVectorizer._vectorize(tokenized, self.encoder.stoi)
        loader = LanguageModelLoader(vectorized, bs=64, bptt=70)

        optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=1e-3,
            betas=(0.8, 0.99),
            weight_decay=1e-7,
        )

        total_loss = 0.
        for i, (data, target) in tqdm.tqdm(enumerate(loader), total=len(loader)):
            optimizer.zero_grad()
            output = self.encoder(Variable(data))
            loss = F.cross_entropy(output, Variable(target))
            loss.backward()
            total_loss += loss.data[0]
            optimizer.step()
        return total_loss / len(x_texts)

    def evaluate(self, x, y):
        return 0., 0.

    def forward(self, x):
        x = self.encoder.encoder(x)
        l_x = self.decoder(x)[0]
        return l_x

    def get_layer_groups(self):
        return [
            (self.encoder.encoder, self.encoder.dropouti),
            *zip(self.encoder.rnns, self.encoder.dropouths),
            (self.decoder)
        ]

    def reset(self):
        _ = [c.reset() for c in self.children() if hasattr(c, 'reset')]

    def score(self, x_texts):
        self.eval()
        tokenized = Tokenizer().proc_all(x_texts)
        x = TextVectorizer.vectorize(tokenized, self.encoder.stoi)
        dataloader = DataLoader(
            dataset=RaggedDataset(x, y=torch.zeros(len(x)).long() - 1),
            batch_size=32,
            collate_fn=text_collate_fn,
            shuffle=False,
            num_workers=1,
        )
        all_outputs = []
        for i, (data, _) in enumerate(dataloader):
            output = self.forward(Variable(data))
            all_outputs.append(output)
        return torch.cat(all_outputs).data.numpy()

    def fit(self, x_texts, y_train):
        tokenized = Tokenizer().proc_all(x_texts)
        X_train = TextVectorizer.vectorize(tokenized, self.encoder.stoi)

        dataloader = DataLoader(
            dataset=RaggedDataset(X_train, y_train),
            sampler=SortishSampler(X_train, batch_size=32),
            batch_size=32,
            collate_fn=text_collate_fn,
            num_workers=1,
            pin_memory=True,
        )

        total_loss = 0.
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            x_batch = Variable(data)
            y_batch = Variable(target)
            output = self.forward(x_batch)
            loss = F.cross_entropy(output, y_batch)
            total_loss += loss.data[0]

        return (total_loss / len(x_texts), 0)

def main():
    pass
    ##n_tok = len(pickle.load(open('/Users/chris_zhu/Documents/Github/ulm-basenet/runs/1/itos.pkl', 'rb')))
    ##
    ##lm = LanguageModel(
    ##    n_tok = n_tok,
    ##)
    ##lm_weights, n_tok = load_lm_weights(
    ##    '/Users/chris_zhu/Documents/Github/ulm-basenet/models/wt103/fwd_wt103.h5',
    ##    '/Users/chris_zhu/Documents/Github/ulm-basenet/models/wt103/itos_wt103.pkl',
    ##    '/Users/chris_zhu/Documents/Github/ulm-basenet/runs/1/itos.pkl',
    ##)
    ##lm.load_weights(lm_weights)

    ##x_valid = np.load('/Users/chris_zhu/Documents/Github/ulm-basenet/runs/1/lm/valid-X.npy')
    ##x = np.concatenate(x_valid)
    ##classifier = LM_TextClassifier(
    ##    language_model=lm,
    ##    n_classes=2,
    ##    predict_only=False
    ##)
    ###classifier.representation_learning(x)
    ##X_train = np.load('/Users/chris_zhu/Documents/Github/ulm-basenet/runs/1/classifier/train-X.npy')
    ##y_train = np.load('/Users/chris_zhu/Documents/Github/ulm-basenet/runs/1/classifier/train-y.npy')
    ##ulabs = np.unique(y_train)
    ##n_class = len(ulabs)
    ##lab_lookup = dict(zip(ulabs, range(len(ulabs))))
    ##y_train = np.array([lab_lookup[l] for l in y_train])
    ##classifier.train(X_train, y_train)


if __name__ == "__main__":
    main()

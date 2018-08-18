import torch
import pickle
import numpy as np
from torch.autograd import Variable

from dataset import Dataset
from label import Label
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from models.sequence_model import SequenceModel
from models.stub_model import StubModel
from models.tf_pretrained_model import TFPretrainedModel
from models.lightnet_model import LightnetModel
from keras.models import load_model
#from models.language_model import LM_TextClassifier, LanguageModel
from utils.text_utils import Tokenizer, UNKNOWN_TOKEN, EOS_TOKEN, PAD_TOKEN

from nltk.corpus import words


def flatten(l):
    return [item for sublist in l for item in sublist]

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

def load_lm_weights(
        ds_id2word,
        lm_id2word_path='vendor/keras_langauge_model.h5',
        lm_model_path='vendor/keras_langauge_model.pkl',
    ):
    load_model()

# Copy all encoder weights, and add our own weights
def load_lm_weights(
    ds_itos,
    lm_weights_path='/Users/chris_zhu/Documents/Github/ulm-basenet/models/wt103/fwd_wt103.h5',
    lm_itos_path='/Users/chris_zhu/Documents/Github/ulm-basenet/models/wt103/itos_wt103.pkl',
):
    lm_weights = torch.load(lm_weights_path, map_location=lambda storage, loc: storage)
    lm_itos = pickle.load(open(lm_itos_path, 'rb'))
    lm_stoi = {v:k for k,v in enumerate(lm_itos)}

    ds_stoi = {v:k for k,v in enumerate(ds_itos)}
    lm_words = set(lm_stoi.keys())
    ds_words = set(ds_stoi.keys())
    all_words = ds_words.union(lm_words)
    n_tok = len(all_words)

    # Adjust vocabulary to match finetuning corpus
    lm_enc_weights = to_numpy(lm_weights['0.encoder.weight'])

    tmp = np.zeros((n_tok, lm_enc_weights.shape[1]), dtype=np.float32)
    tmp += lm_enc_weights.mean(axis=0)

    all_itos = list(all_words)
    for i, word in enumerate(all_words):
        if word in lm_stoi:
            tmp[i] = lm_enc_weights[lm_stoi[word]]

    lm_weights['0.encoder.weight']                    = torch.Tensor(tmp.copy())
    lm_weights['0.encoder_with_dropout.embed.weight'] = torch.Tensor(tmp.copy())
    lm_weights['1.decoder.weight']                    = torch.Tensor(tmp.copy())

    return lm_weights, all_itos


class ModelBuilder:
    def __init__(self, dataset, label, config):
        self.config = config
        self.dataset = dataset
        self.label = label

    def build_custom_model(self):
        return TFPretrainedModel(self.config['module'], self.config['directory'])

    def build_language_model(self):
        """Build base model for seq2seq, text classification, and sequence labelling."""
        data = self.dataset.all
        assert 'text' in data.columns, "Not a text dataset, can't build language model"
        texts = data['text'].values
        tokenized = Tokenizer().proc_all(texts)
        vocab = set(flatten(tokenized))
        lm_weights, itos = load_lm_weights(vocab)

        lm = LanguageModel(itos)
        lm.load_weights(lm_weights)
        return lm

    def build(self):
        if 'custom' in self.config and self.config['custom']:
            return self.build_custom_model()

        #if self.dataset.data_type == Dataset.TEXT_TYPE:
        #    lm = self.build_language_model()

        # Right now there is an assumption that
        if self.dataset.data_type == Dataset.IMAGE_TYPE and self.label.label_type == Label.BINARY:
            return CNNModel(2, input_shape=(128, 128))

        if self.dataset.data_type == Dataset.IMAGE_TYPE and self.label.label_type == Label.CLASSIFICATION:
            return CNNModel(len(self.label.classes), input_shape=(128, 128))

        word_list = [word.lower() for word in words.words()] + [UNKNOWN_TOKEN, EOS_TOKEN, PAD_TOKEN]

        if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.BINARY:
            return RNNModel(2, word_list)

        if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.CLASSIFICATION:
            return RNNModel(len(self.label.classes), word_list)
            #return LM_TextClassifier(lm, len(self.label.classes))

        if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.SEQUENCE:
            return SequenceModel(
                valid_outputs=self.label.valid_tokens,
                seq2seq=False,
                character_mode=False,
            )

        if self.dataset.data_type == Dataset.OBJECT_DETECTION_TYPE and self.label.label_type == Label.OBJECT_DETECTION:
            return LightnetModel()

        return StubModel()

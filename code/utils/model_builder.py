import pdb
import pickle
import numpy as np

from dataset import Dataset
from label import Label
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from models.cnn_text_classifier import CNNTextClassifier
from models.sequence_model import SequenceModel
from models.stub_model import StubModel
from models.tf_pretrained_model import TFPretrainedModel
from keras.models import load_model
#from models.language_model import LM_TextClassifier, LanguageModel
from utils.text_utils import Tokenizer, UNKNOWN_TOKEN, EOS_TOKEN, PAD_TOKEN


def flatten(l):
    return [item for sublist in l for item in sublist]

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

        word_list = pickle.load(open("vendor/keras_language_model/vocab.p", "rb"))

        if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.BINARY:
            cnn_model = CNNTextClassifier(2)
            return cnn_model
            #rnn_model = RNNModel(2, word_list)
            #rnn_model.load_lm('vendor/keras_language_model')
            #return rnn_model

        if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.CLASSIFICATION:
            cnn_model = CNNTextClassifier(len(self.label.classes))
            return cnn_model
            #rnn_model = RNNModel(len(self.label.classes), word_list)
            #rnn_model.load_lm('vendor/keras_language_model')
            #return rnn_model
            #return LM_TextClassifier(lm, len(self.label.classes))

        if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.SEQUENCE:
            return SequenceModel(
                valid_outputs=self.label.valid_tokens,
                seq2seq=False,
                character_mode=False,
            )

        if self.dataset.data_type == Dataset.OBJECT_DETECTION_TYPE and self.label.label_type == Label.OBJECT_DETECTION:
            from models.lightnet_model import LightnetModel
            return LightnetModel()

        return StubModel()

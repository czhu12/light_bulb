import pdb
import pickle
import numpy as np
import logging

from dataset import Dataset
from labels.label import Label
from models.stub_model import StubModel
#from models.language_model import LM_TextClassifier, LanguageModel
#from models.sequence_model import SequenceModel

logger = logging.getLogger('label_app')
logger.setLevel(logging.DEBUG)


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
        built_model = self._build()
        logger.debug("Loading model: {}".format(type(built_model).__name__))
        return built_model

    def _build(self):
        if 'custom' in self.config and self.config['custom']:
            return self.build_custom_model()

        #if self.dataset.data_type == Dataset.TEXT_TYPE:
        #    lm = self.build_language_model()

        # Right now there is an assumption that
        #if self.dataset.data_type == Dataset.IMAGE_TYPE and self.label.label_type == Label.BINARY:
        #    return CNNModel(2, input_shape=(128, 128))

        #if self.dataset.data_type == Dataset.IMAGE_TYPE and self.label.label_type == Label.CLASSIFICATION:
        #    return CNNModel(len(self.label.classes), input_shape=(128, 128))

        #word_list = pickle.load(open("vendor/keras_language_model/vocab.p", "rb"))

        #if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.BINARY:
        #    cnn_model = CNNTextClassifier(2)
        #    return cnn_model

        #if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.CLASSIFICATION:
        #    cnn_model = CNNTextClassifier(len(self.label.classes))
        #    return cnn_model

        #if self.dataset.data_type == Dataset.JSON_TYPE and self.label.label_type == Label.SEQUENCE:
        #    from models.cnn_sequence_tagger import CNNSequenceTagger
        #    return CNNSequenceTagger(self.label.score_classes)

        #if self.dataset.data_type == Dataset.OBJECT_DETECTION_TYPE and self.label.label_type == Label.OBJECT_DETECTION:
        #    from models.lightnet_model import LightnetModel
        #    return LightnetModel()

        return StubModel()

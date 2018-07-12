from dataset import Dataset
from label import Label
from models.cnn_model import CNNModel
from models.rnn_model import RNNModel
from models.sequence_model import SequenceModel
from models.stub_model import StubModel
from models.tf_pretrained_model import TFPretrainedModel

class ModelBuilder:
    def __init__(self, dataset, label, config):
        self.config = config
        self.dataset = dataset
        self.label = label

    def build_custom_model(self):
        return TFPretrainedModel(self.config['module'], self.config['directory'])

    def build(self):
        if 'custom' in self.config and self.config['custom']:
            return self.build_custom_model()
        # Right now there is an assumption that
        if self.dataset.data_type == Dataset.IMAGE_TYPE and self.label.label_type == Label.BINARY:
            return CNNModel(input_shape=(128, 128))
        if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.BINARY:
            return RNNModel()
        if self.dataset.data_type == Dataset.TEXT_TYPE and self.label.label_type == Label.SEQUENCE:
            return SequenceModel(
                valid_outputs=self.label.valid_tokens,
                seq2seq=False,
                character_mode=False,
            )

        return StubModel()

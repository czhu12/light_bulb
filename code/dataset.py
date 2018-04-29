import pandas as pd
import os
import glob
from utils import utils
import chardet
from models.cnn_models import CNNModel
from models.rnn_model import RNNModel
from utils import glove_utils


class Dataset:
    TRAIN = 'TRAIN'
    TEST = 'TEST'
    LABEL_MAP = { 'YES': 1, 'NO': 0 }
    IMAGE_TYPE = 'images'
    TEXT_TYPE = 'text'
    MIN_TRAIN_EXAMPLES = 20 
    MIN_TEST_EXAMPLES = 20
    MIN_UNSUPERVISED_EXAMPLES = 100

    @staticmethod
    def load_from(config):
        if config['data_type'] == Dataset.IMAGE_TYPE:
            return ImageDataset(config)
        else:
            return TextDataset(config)

    def __init__(self, config):
        self.data_type = config.get('data_type')
        self.directory = config.get('directory')
        self.judgements_file = config.get('judgements_file')

        # Start dataset with existing judgements
        self.dataset = self.load_existing_judgements()
        # Add unlabelled examples
        self.dataset = self.dataset.append(self.load_unlabelled_dataset())
        self.loaded_text = False
        self.current_stage = Dataset.TRAIN

    def save(self):
        self.labelled.to_csv(self.judgements_file, index=False)

    def load_unlabelled_dataset(self):
        raise NotImplementedError

    def empty_dataframe(self):
        df = pd.DataFrame(columns=self.__class__.COLUMNS)
        return df

    def load_existing_judgements(self):
        if not os.path.isfile(self.judgements_file):
            return self.empty_dataframe()

        df = pd.read_csv(self.judgements_file)
        assert sorted(df.columns) == sorted(self.__class__.COLUMNS)
        df['labelled'] = True
        return df

    @property
    def unlabelled(self):
        return self.dataset[self.dataset['labelled'] == False]

    @property
    def labelled(self):
        return self.dataset[self.dataset['labelled'] == True]

    @property
    def train_data(self):
        return self.labelled[self.labelled['stage'] == Dataset.TRAIN]

    @property
    def test_data(self):
        return self.labelled[self.labelled['stage'] == Dataset.TEST]

    def sample(self, size=100):
        if len(self.unlabelled) == 0:
            return self.unlabelled
        return self.unlabelled.sample(size)

    @property
    def model(self):
        raise NotImplementedError

    @property
    def train_set(self):
        raise NotImplementedError

    @property
    def unlabelled_set(self):
        raise NotImplementedError

    @property
    def ready_to_evaluate(self):
        return len(self.test_data) > Dataset.MIN_TEST_EXAMPLES

    @property
    def ready_to_train(self):
        return len(self.train_data) > Dataset.MIN_TRAIN_EXAMPLES

    def set_current_stage(self):
        if len(self.test_data) <= Dataset.MIN_TEST_EXAMPLES:
            self.current_stage = Dataset.TEST

        ratio = len(self.test_data) / (len(self.train_data) + 1)
        
        if ratio < 0.33:
            self.current_stage = Dataset.TEST
        else:
            self.current_stage = Dataset.TRAIN

    @property
    def test_set(self):
        raise NotImplementedError

    def add_label(self, id, label, stage):
        self.dataset.loc[self.dataset['path'] == id, ['label', 'labelled', 'stage']] = [label, True, stage]
        self.save()

class TextDataset(Dataset):
    COLUMNS = ['label', 'labelled_by', 'path', 'labelled', 'stage', 'text']

    def load_unlabelled_dataset(self):
        types = ['*.txt']
        text_paths = []
        for type in types:
            text_paths.extend(glob.glob("{}/{}".format(self.directory, type)))

        unlabelled_paths = set(text_paths) - set(self.dataset['path'].values)
        data = []
        for path in unlabelled_paths:
            data.append({
                'path': path,
                'labelled': False,
                'text': open(path).read(),
            })
        new_dataset = pd.DataFrame(
            data,
            columns=self.__class__.COLUMNS,
        )
        return new_dataset

    @property
    def model(self):
        return RNNModel(self.dataset['text'].values)
    
    @property
    def test_set(self):
        if len(self.test_data) == 0:
            return [], []

        test_data = self.test_data
        x_train = test_data['text'].values
        y_train = utils.one_hot_encode(test_data['label'].values)
        return x_train, y_train

    @property
    def train_set(self):
        if len(self.train_data) == 0:
            return [], []
        train_data = self.train_data
        x_train = train_data['text'].values
        y_train = utils.one_hot_encode(train_data['label'].values)
        return x_train, y_train

    def unlabelled_set(self, size=Dataset.MIN_UNSUPERVISED_EXAMPLES):
        if len(self.unlabelled) < Dataset.MIN_UNSUPERVISED_EXAMPLES:
            raise ValueError("Need at least {} examples for unsupervised training.".format(Dataset.MIN_UNSUPERVISED_EXAMPLES))

        data = self.sample(size)
        return data['text'].values, data['path'].values

class ImageDataset(Dataset):
    COLUMNS = ['label', 'labelled_by', 'path', 'labelled', 'stage']

    def __init__(self, config):
        super(ImageDataset, self).__init__(config)
        if 'image_width' in config and 'image_height' in config:
            self.input_shape = (int(config['image_width']), int(config['image_height']))
        else:
            self.input_shape = (128, 128)

    def load_unlabelled_dataset(self):
        types = ['*.jpg', '*.png']
        image_paths = []
        for type in types:
            image_paths.extend(glob.glob("{}/{}".format(self.directory, type)))

        unlabelled_paths = set(image_paths) - set(self.dataset['path'].values)
        new_dataset = pd.DataFrame([{'path': path, 'labelled': False} for path in unlabelled_paths], columns=self.__class__.COLUMNS)
        return new_dataset

    @property
    def test_set(self):
        if len(self.test_data) == 0:
            return [], []
        test_data = self.test_data
        x_train = utils.load_images(test_data['path'].values, self.input_shape)
        y_train = utils.one_hot_encode(test_data['label'].values)
        return x_train, y_train

    @property
    def train_set(self):
        if len(self.train_data) == 0:
            return [], []
        train_data = self.train_data
        x_train = utils.load_images(train_data['path'].values, self.input_shape)
        y_train = utils.one_hot_encode(train_data['label'].values)
        return x_train, y_train

    def unlabelled_set(self, size=Dataset.MIN_UNSUPERVISED_EXAMPLES):
        if len(self.unlabelled) < Dataset.MIN_UNSUPERVISED_EXAMPLES:
            raise ValueError("Need at least {} examples for unsupervised training.".format(Dataset.MIN_UNSUPERVISED_EXAMPLES))
        data = self.sample(size)
        x_train = utils.load_images(data['path'].values, self.input_shape)
        return x_train, data['path'].values

    @property
    def model(self):
        return CNNModel(input_shape=self.input_shape)

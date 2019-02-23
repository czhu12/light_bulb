import pandas as pd
import os
import glob
from utils import utils
import chardet
from utils import text_utils
from threading import Lock
import re

import copy
import logging
import json
from PIL import Image
from typing import List

MIN_TRAIN_EXAMPLES = 40
MIN_TEST_EXAMPLES = 10
MIN_UNSUPERVISED_EXAMPLES = 100


class Dataset:
    TRAIN = 'TRAIN'
    TEST = 'TEST'
    MODEL_LABELLED = 'MODEL_LABELLED'
    USER_MODEL_DISAGREEMENT = 'USER_MODEL_DISAGREEMENT'
    USER_MODEL_LABELLER = 'USER_MODEL_LABELLER'
    JSON_TYPE = 'json'
    IMAGE_TYPE = 'images'
    TEXT_TYPE = 'text'
    OBJECT_DETECTION_TYPE = 'object_detection'

    @staticmethod
    def load_from(config):
        if config['data_type'] == Dataset.IMAGE_TYPE:
            return ImageDataset(config)
        elif config['data_type'] == Dataset.TEXT_TYPE:
            return TextDataset(config)
        elif config['data_type'] == Dataset.JSON_TYPE:
            return JSONDataset(config)
        elif config['data_type'] == Dataset.OBJECT_DETECTION_TYPE:
            return ObjectDetectionDataset(config)
        else:
            pass

    def __init__(self, config):
        self.data_type = config.get('data_type')
        self.directory = config.get('directory')
        self.judgements_file = config.get('judgements_file')
        self.save_lock = Lock()

        # Start dataset with existing judgements
        self.dataset = self.load_existing_judgements()
        # Add unlabelled examples
        unlabelled = self.load_unlabelled_dataset()

        self.dataset = self.dataset.append(unlabelled)
        self.loaded_text = False
        self.current_stage = Dataset.TEST

    def search(self, search_query):
        raise NotImplementedError()

    def save(self):
        self.save_lock.acquire()
        try:
            self.labelled.to_csv(self.judgements_file, index=False)
        finally:
            self.save_lock.release()

    @property
    def all(self):
        return self.dataset

    def model_labelled(self, num=100):
        model_labelled = self.dataset[self.dataset['stage'] == Dataset.MODEL_LABELLED]
        if len(model_labelled) == 0:
            return self.empty_dataframe(), -1
        target_class = model_labelled['label'].mode()[0]
        model_labelled = model_labelled[model_labelled['label'] == target_class]
        if len(model_labelled) < num:
            return model_labelled, int(target_class)
        return model_labelled.sample(num), int(target_class)

    @property
    def stats(self):
        return {
            "labelled": {
                "total": len(self.labelled),
                "train": len(self.train_data),
                "model_labelled": len(self.model_labelled_data),
                "test": len(self.test_data),
            },
            "unlabelled": len(self.unlabelled),
            "average_time_taken": self.labelled['time_taken'].median() if len(self.labelled) > 0 else 0,
        }

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
    def model_label(self):
        return self.dataset[(self.dataset['labelled'] == False) & (self.dataset['stage'] != Dataset.MODEL_LABELLED)]

    @property
    def labelled(self):
        return self.dataset[self.dataset['labelled'] == True]

    @property
    def model_labelled_data(self):
        return self.dataset[self.dataset['stage'] == Dataset.MODEL_LABELLED]

    @property
    def train_data(self):
        return self.labelled[self.labelled['stage'] == Dataset.TRAIN]

    @property
    def test_data(self):
        return self.labelled[self.labelled['stage'] == Dataset.TEST]

    def sample(self, size=100):
        if len(self.unlabelled) < size:
            return self.unlabelled
        return self.unlabelled.sample(size)

    @property
    def train_set(self):
        raise NotImplementedError

    def unlabelled_set(self):
        raise NotImplementedError

    def model_labelling_set(self):
        raise NotImplementedError

    def set_current_stage(self):
        if len(self.test_data) <= MIN_TEST_EXAMPLES:
            self.current_stage = Dataset.TEST
            return

        ratio = len(self.test_data) / (len(self.train_data) + 1)

        if ratio < 0.33:
            self.current_stage = Dataset.TEST
        else:
            self.current_stage = Dataset.TRAIN

    @property
    def test_set(self):
        raise NotImplementedError

    def add_label(self, id, label, stage, user='default', save=True, is_labelled=True, time_taken=0.):
        self.dataset.loc[self.dataset['path'] == id, [
            'label',
            'labelled',
            'stage',
            'labelled_by',
            'time_taken',
        ]] = [label, is_labelled, stage, user, time_taken]
        if save:
            self.save()

    def get_data(self, _id):
        return None

class TextDataset(Dataset):
    COLUMNS = ['label', 'labelled_by', 'path', 'labelled', 'stage', 'text', 'time_taken']

    def get_data(self, _id):
        return self.dataset[self.dataset['path'] == _id]['text'].values[0]

    def search(self, search_query, num_results):
        text = self.unlabelled['text']
        text = ' ' + text + ' '
        text = text.str.lower().str.replace('[^\w\s]', ' ')
        search_query = search_query.lower()
        search_query = re.sub('[^\w\s]', ' ', search_query)
        return self.unlabelled[
            text.str.contains(' ' + search_query + ' ')
        ].iloc[:num_results]

    def load_unlabelled_dataset(self, types=['*.txt']):
        text_paths = []
        for type in types:
            text_paths.extend(glob.glob("{}/{}".format(self.directory, type)))

        if len(set(text_paths)) == 0:
            raise ValueError("No txt files found in {}. Probably pointing at a wrong directory?".format(self.directory))

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
    def test_set(self):
        if len(self.test_data) == 0:
            return [], []

        test_data = self.test_data
        x_train = test_data['text'].values
        y_train = test_data['label'].values
        return x_train, y_train

    @property
    def train_set(self):
        if len(self.train_data) == 0:
            return [], []
        train_data = self.train_data
        x_train = train_data['text'].values
        y_train = train_data['label'].values
        return x_train, y_train

    def unlabelled_set(self, size=MIN_UNSUPERVISED_EXAMPLES):
        data = self.sample(size)
        if len(data) > 0:
            x_train = data['text'].values
            ids = data['path'].values
        else:
            x_train = []
            ids = []
        return x_train, ids

    def model_labelling_set(self, size=MIN_UNSUPERVISED_EXAMPLES):
        data = self.model_label if len(self.model_label) < size else self.model_label.sample(size)
        if len(data) > 0:
            x_train = data['text'].values
            ids = data['path'].values
        else:
            x_train = []
            ids = []
        return x_train, ids

class JSONDataset(TextDataset):

    def load_unlabelled_dataset(self):
        types = ['*.json']
        return super(JSONDataset, self).load_unlabelled_dataset(types)

    @property
    def test_set(self):
        if len(self.test_data) == 0:
            return [], []

        test_data = self.test_data
        x_train = [json.loads(val) for val in test_data['text'].values]
        y_train = test_data['label'].values
        return x_train, y_train

    @property
    def train_set(self):
        if len(self.train_data) == 0:
            return [], []
        train_data = self.train_data
        x_train = [json.loads(val) for val in train_data['text'].values]
        y_train = train_data['label'].values
        return x_train, y_train

    def unlabelled_set(self, size=MIN_UNSUPERVISED_EXAMPLES):
        data = self.sample(size)
        if len(data) > 0:
            x_train = [json.loads(val) for val in data['text'].values]
            ids = data['path'].values
        else:
            x_train = []
            ids = []
        return x_train, ids

    def model_labelling_set(self, size=MIN_UNSUPERVISED_EXAMPLES):
        data = self.model_label if len(self.model_label) < size else self.model_label.sample(size)
        if len(data) > 0:
            x_train = [json.loads(val) for val in data['text'].values]
            ids = data['path'].values
        else:
            x_train = []
            ids = []
        return x_train, ids

    def model_labelled(self, num=100):
        model_labelled = self.dataset[self.dataset['stage'] == Dataset.MODEL_LABELLED]
        if len(model_labelled) == 0:
            return -1, self.empty_dataframe()
        return model_labelled.sample(min(num, len(model_labelled))), -1

class ImageDataset(Dataset):
    COLUMNS = ['label', 'labelled_by', 'path', 'labelled', 'stage', 'time_taken']

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

        if len(set(image_paths)) == 0:
            raise ValueError(
                "No images (jpg or png) files found in {}.".format(self.directory),
            )
        unlabelled_paths = set(image_paths) - set(self.dataset['path'].values)
        new_dataset = pd.DataFrame([{'path': path, 'labelled': False} for path in unlabelled_paths], columns=self.__class__.COLUMNS)
        return new_dataset

    @property
    def test_set(self):
        if len(self.test_data) == 0:
            return [], []
        test_data = self.test_data
        x_train = utils.load_images(test_data['path'].values, self.input_shape)
        y_train = test_data['label'].values
        return x_train, y_train

    @property
    def train_set(self):
        if len(self.train_data) == 0:
            return [], []
        train_data = self.train_data
        x_train = utils.load_images(train_data['path'].values, self.input_shape)
        y_train = train_data['label'].values
        return x_train, y_train

    def unlabelled_set(self, size=MIN_UNSUPERVISED_EXAMPLES):
        data = self.sample(size)
        if len(data) > 0:
            x_train = utils.load_images(data['path'].values, self.input_shape)
            ids = data['path'].values
        else:
            x_train = []
            ids = []
        return x_train, ids

    def model_labelling_set(self, size=MIN_UNSUPERVISED_EXAMPLES):
        data = self.model_label if len(self.model_label) < size else self.model_label.sample(size)
        if len(data) > 0:
            x_train = utils.load_images(data['path'].values, self.input_shape)
            ids = data['path'].values
        else:
            x_train = []
            ids = []
        return x_train, ids

class ObjectDetectionDataset(ImageDataset):
    # Object detection
    def __init__(self):
        super(ObjectDetectionDataset, self).__init__()

        from brambox.boxes.annotations import Annotation
        import lightnet.data as lnd

    def serialize_brambox(self, box) -> str:
        return json.dumps({
            'class_label': box.class_label,
            'object_id': box.object_id,
            'x_top_left': box.x_top_left,
            'y_top_left': box.y_top_left,
            'width': box.width,
            'height': box.height,
        })

    #def _deserialize_bramboxes(self, encoding: str) -> List[Annotation]:
    def _deserialize_bramboxes(self, encoding: str):
        boxes = json.loads(encoding)
        return [self._deserialize_brambox(box) for box in boxes]

    #def _deserialize_brambox(self, obj: str) -> Annotation:
    def _deserialize_brambox(self, obj: str):
        box = Annotation()
        box.class_label = obj['class_label']
        box.object_id = 0
        box.x_top_left = obj['x_top_left']
        box.y_top_left = obj['y_top_left']
        box.width = obj['width']
        box.height = obj['height']
        return box

    @property
    def test_set(self):
        if len(self.test_data) == 0:
            return [], []
        test_data = self.test_data
        paths = train_data['path'].values
        boxes = [self._deserialize_bramboxes(label) for label in train_data['label'].values]
        return paths, boxes

    @property
    def train_set(self):
        if len(self.train_data) == 0:
            return [], []
        train_data = self.train_data
        paths = train_data['path'].values
        boxes = [deserialize_bramboxes(label) for label in train_data['label'].values]
        return paths, boxes

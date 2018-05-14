from queue import Queue
import numpy as np
import yaml
import time

import pdb

from utils import utils
from utils import model_builder
import logging
import sys
from dataset import Dataset
from training.trainer import Trainer
from labeller import ModelLabeller
from label import Label
from operator import itemgetter
CLASSIFICATION_COLORS = [
"#2ecc71",
"#9b59b6",
"#f1c40f",
"#e74c3c",
"#16a085",
"#27ae60",
"#2980b9",
"#8e44ad",
"#2c3e50",
"#f39c12",
"#d35400",
"#c0392b",
"#1abc9c",
"#3498db",
"#34495e",
"#e67e22",
]

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

class LabelApp:
    @staticmethod
    def load_from(config):
        with open(config) as f:
            config = yaml.load(f)

        task = Task.load_from(config['task'])
        dataset = Dataset.load_from(config['dataset'])
        model_directory = config['model_directory']
        label_helper = Label.load_from(config['label'])
        user = config['user']

        return LabelApp(task, dataset, model_directory, label_helper, user)

    def __init__(self, task, dataset, model_directory, label_helper, user, model_labelling=True):
        self.task = task
        self.dataset = dataset
        self.data_type = self.dataset.data_type

        self.label_helper = label_helper
        self.label_type = label_helper.label_type

        self.model = model_builder.ModelBuilder(dataset, self.label_helper).build()

        self.trainer = Trainer(model_directory, self.model, self.dataset, logger=logger)
        self.trainer.load_existing()

        self.labeller = ModelLabeller(self.model, self.dataset, logger=logger)

        self.user = user
        self.model_labelling = model_labelling

    def score(self, x):
        scores = self.model.score(x)
        return scores

    def predict(self, x):
        predictions = self.model.predict(x)
        return predictions

    def next_batch(self, size=10):
        logger.debug("Sampling a batch for {} set.".format(self.dataset.current_stage))
        self.dataset.set_current_stage()
        if self.dataset.current_stage == Dataset.TEST:
            sampled_df = self.dataset.sample(size)
            return sampled_df, 0, self.dataset.current_stage, [] # TODO: This needs to be fixed

        # Generate training data
        sampled_df = self.dataset.sample(size * 5)
        if self.data_type == Dataset.IMAGE_TYPE:
            x_data = utils.load_images(sampled_df['path'].values, self.dataset.input_shape)
        if self.data_type == Dataset.TEXT_TYPE:
            x_data = sampled_df['text'].values

        scores = self.model.score(x_data)
        entropy = np.sum(scores * np.log(scores) / np.log(2), axis=-1)
        if len(entropy.shape) > 1:
            entropy = entropy.mean(1)
        max_entropy_indexes = np.argpartition(-entropy, size)[:size]
        response = (
            sampled_df.iloc[max_entropy_indexes],
            max_entropy_indexes.tolist(),
            self.dataset.current_stage,
            x_data[max_entropy_indexes],
        )
        return response


    def add_label(self, _id, label):
        # Validate label
        # TODO: Reevaluate this get_data thing, I'm not a fan of this.
        data = self.dataset.get_data(_id)
        self.label_helper.validate(data, label)
        label = self.label_helper.decode(label)
        # _id is just the path to the file
        self.dataset.add_label(_id, label, self.dataset.current_stage, user=self.user)

    @property
    def title(self):
        return self.task.title

    @property
    def description(self):
        return self.task.description

    def threaded_train(self):
        self.trainer.train()

    def threaded_label(self):
        self.labeller.start()

    def get_history(self):
        return self.trainer.get_history()

    def get_stats(self):
        return self.dataset.stats

class Task:
    @staticmethod
    def load_from(config):
        return Task(**config)

    def __init__(self, title='', description=''):
        self.title = title
        self.description = description

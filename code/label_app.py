from queue import Queue
import numpy as np
import yaml
import time

import pdb

from utils import utils
import logging
import sys
from dataset import Dataset
from training.trainer import Trainer
from labeller import ModelLabeller

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
        return LabelApp(task, dataset, model_directory)

    def __init__(self, task, dataset, model_directory):
        self.task = task
        self.dataset = dataset
        self.data_type = self.dataset.data_type
        self.model = self.dataset.model

        self.trainer = Trainer(model_directory, self.model, self.dataset, logger=logger)
        self.trainer.load_existing()

        self.labeller = ModelLabeller(self.model, self.dataset, logger=logger)

    def score(self, x_train):
        scores = self.model.score(x_train)
        return scores

    def next_batch(self, size=10):
        logger.debug("Sampling a batch for {} set.".format(self.dataset.current_stage))
        self.dataset.set_current_stage()
        if self.dataset.current_stage == Dataset.TEST:
            sampled_df = self.dataset.sample(size)
            return sampled_df, 0, self.dataset.current_stage

        # Generate training data
        sampled_df = self.dataset.sample(size * 5)
        if self.data_type == Dataset.IMAGE_TYPE:
            x_data = utils.load_images(sampled_df['path'].values, self.dataset.input_shape)
        if self.data_type == Dataset.TEXT_TYPE:
            x_data = sampled_df['text'].values

        scores = self.model.score(x_data)
        entropy = np.sum(scores * np.log(scores) / np.log(2), axis=1)
        max_entropy_indexes = np.argpartition(-entropy, size)[:size]
        return sampled_df.iloc[max_entropy_indexes], max_entropy_indexes.tolist(), self.dataset.current_stage

    def add_label(self, _id, label):
        # _id is just the path to the file
        if label not in Dataset.LABEL_MAP:
            raise ValueError('{} is not a valid label'.format(label))
        label = Dataset.LABEL_MAP[label]
        self.dataset.add_label(_id, label, self.dataset.current_stage)

    @property
    def title(self):
        return self.task.title

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
        return Task(title=config['title'])

    def __init__(self, title=''):
        self.title = title

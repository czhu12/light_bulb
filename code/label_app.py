from queue import Queue
import numpy as np
import yaml
import time
from operator import itemgetter

import pdb

from utils import utils
from utils import model_builder
from utils.config_parser import ConfigParser
import logging
import sys
from dataset import Dataset
from training.trainer import Trainer
from labeller import ModelLabeller
from label import Label

logger = logging.getLogger('label_app')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

class LabelApp:
    @staticmethod
    def load_from(config_path):
        with open(config_path) as f:
            config = yaml.load(f)
            parser = ConfigParser(config)
            parser._create_directories()

        task = Task.load_from(parser.task)
        dataset = Dataset.load_from(parser.dataset)
        model_config = config['model']
        label_helper = Label.load_from(parser.label)
        user = config['user']

        return LabelApp(task, dataset, label_helper, user, model_config, parser)

    def __init__(self, task, dataset, label_helper, user, model_config, config, model_labelling=True):
        self.config = config.config
        self.task = task
        self.dataset = dataset
        self.data_type = self.dataset.data_type

        self.label_helper = label_helper

        model_directory = model_config['directory']
        self.model = model_builder.ModelBuilder(dataset, self.label_helper, model_config).build()

        self.trainer = Trainer(model_directory, self.model, self.dataset, self.label_helper, logger=logger)
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

    @property
    def is_done(self):
        return len(self.dataset.unlabelled) == 0

    def next_batch(self, size=10, force_stage=None, reverse_entropy=False):
        if self.is_done:
            raise ValueError("Tried to sample a batch when there is nothing else to sample")

        logger.debug("Sampling a batch for {} set.".format(self.dataset.current_stage))
        self.dataset.set_current_stage()

        current_stage = force_stage if force_stage else self.dataset.current_stage

        if current_stage == Dataset.TEST:
            sampled_df = self.dataset.sample(size)
            return sampled_df, current_stage, [], [0.5] * len(sampled_df) # TODO: This needs to be fixed

        # Generate training data
        sampled_df = self.dataset.sample(size * 5)
        if self.data_type == Dataset.IMAGE_TYPE:
            x_data = utils.load_images(sampled_df['path'].values, self.dataset.input_shape)
        if self.data_type == Dataset.TEXT_TYPE:
            x_data = sampled_df['text'].values

        scores = self.model.score(x_data)
        entropy_func = lambda scores: np.sum(scores * np.log(1 / scores), axis=-1)
        if type(scores) == list:
            entropy = np.array([entropy_func(score).mean() for score in scores])
        else:
            entropy = entropy_func(scores)

        assert len(entropy.shape) == 1

        num = min(size, len(entropy) - 1)
        if reverse_entropy:
            entropy_indexes = np.argpartition(entropy, num)[:num]
        else:
            entropy_indexes = np.argpartition(-entropy, num)[:num]
        response = (
            sampled_df.iloc[entropy_indexes],
            current_stage,
            x_data[entropy_indexes],
            entropy[entropy_indexes].tolist(),
        )
        return response


    def add_label(self, _id, label):
        # Validate label
        # TODO: Reevaluate this get_data thing, I'm not a fan of this.
        data = self.dataset.get_data(_id)
        self.label_helper.validate(data, label)
        print(label)
        print(label)
        print(label)
        print(label)
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

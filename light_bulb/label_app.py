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
from labelling.labeller import ModelLabeller
from labels.label import Label

class LabelApp:
    @staticmethod
    def load_from(config_meta):
        with open(config_meta['path']) as f:
            config = yaml.load(f)
            parser = ConfigParser(config)
            parser._create_directories()

        task = Task.load_from(parser.task)
        dataset = Dataset.load_from(parser.dataset)
        model_config = config['model']
        label_helper = Label.load_from(parser.label)
        user = config['user']

        # Set up logger
        log_level = config_meta['log_level']
        logger = logging.getLogger('label_app')
        logger.setLevel(getattr(logging, log_level))

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

        return LabelApp(task, dataset, label_helper, user, model_config, parser, logger)

    def __init__(self, task, dataset, label_helper, user, model_config, config, logger, model_labelling=True):
        self.config = config.config
        self.task = task
        self.dataset = dataset
        self.data_type = self.dataset.data_type

        self.label_helper = label_helper

        model_directory = model_config['directory']
        self.model = model_builder.ModelBuilder(dataset, self.label_helper, model_config).build()

        self.trainer = Trainer(model_directory, self.model, self.dataset, self.label_helper, logger=logger)
        self.trainer.load_existing()

        self.labeller = ModelLabeller(self.model, self.dataset, self.label_helper, logger=logger)

        self.user = user
        self.model_labelling = model_labelling
        self.logger = logger

    def score(self, x):
        scores = self.model.score(x)
        return scores

    def predict(self, x):
        predictions = self.model.predict(x)
        return predictions

    @property
    def is_done(self):
        return len(self.dataset.unlabelled) == 0

    def next_model_labelled_batch(self, size=100):
        model_labelled, target_class = self.dataset.model_labelled(size)
        return model_labelled, target_class

    def next_batch(self, size=10, force_stage=None, reverse_entropy=False, prediction=False):
        if self.is_done:
            raise ValueError("Tried to sample a batch when there is nothing else to sample")

        self.logger.debug("Sampling a batch for {} set.".format(self.dataset.current_stage))
        self.dataset.set_current_stage()

        current_stage = force_stage if force_stage else self.dataset.current_stage

        if current_stage == Dataset.TEST:
            sampled_df = self.dataset.sample(size)
            return sampled_df, current_stage, [], [0.5] * len(sampled_df) # TODO: This needs to be fixed

        # Generate training data
        sampled_df = self.dataset.sample(size * 5)
        if self.data_type == Dataset.IMAGE_TYPE:
            x_data, ids = self.dataset.unlabelled_set(size * 5)
        if self.data_type == Dataset.TEXT_TYPE:
            x_data, ids = self.dataset.unlabelled_set(size * 5)
        if self.data_type == Dataset.JSON_TYPE:
            x_data, ids = self.dataset.unlabelled_set(size * 5)

        scores = self.model.score(x_data)
        entropy_func = lambda scores: np.sum(scores * np.log(1 / scores), axis=-1)
        if len(scores.shape) == 3:
            entropy = np.array([entropy_func(score).mean() for score in scores])
        else:
            entropy = entropy_func(scores)

        assert len(entropy.shape) == 1

        num = min(size, len(entropy) - 1)
        if reverse_entropy:
            entropy_indexes = np.argpartition(entropy, num)[:num]
        else:
            entropy_indexes = np.argpartition(-entropy, num)[:num]

        # Make predictions
        # TODO: This doesn't work for text or json types
        if self.data_type == Dataset.IMAGE_TYPE:
            x_to_score = x_data[entropy_indexes]
        else:
            x_to_score = []

        y_prediction = None
        if prediction and len(x_to_score) > 0:
            y_prediction = self.predict(x_to_score)

        response = (
            sampled_df.iloc[entropy_indexes],
            current_stage,
            y_prediction,
            entropy[entropy_indexes].tolist(),
        )
        return response

    def search(self, search_query: str, num_results: int=20):
        results = self.dataset.search(search_query, num_results)
        return results

    def labelled_data(self, start_idx, end_idx, labelled=None):
        df = self.dataset.dataset
        if labelled is not None:
            df = df[df['labelled'] == labelled]

        if start_idx >= len(df):
            return [], True

        rows = df.iloc[start_idx:end_idx]
        return rows, False

    def add_labels(self, labels, avg_time_taken):
        for label in labels:
            self.dataset.add_label(
                label['path'],
                label['label'],
                Dataset.TRAIN,
                user=self.user,
                time_taken=avg_time_taken,
            )

    def add_label(self, _id, label, time_taken):
        # Validate label
        # TODO: Reevaluate this get_data thing, I'm not a fan of this.
        data = self.dataset.get_data(_id)
        if data:
            self.label_helper.validate(data, label)
        label = self.label_helper.decode(label)
        # _id is just the path to the file
        self.dataset.add_label(
            _id,
            label,
            self.dataset.current_stage,
            user=self.user,
            save=True,
            time_taken=time_taken,
        )

    @property
    def title(self):
        return self.task.title

    @property
    def description(self):
        return self.task.description

    @property
    def template(self):
        return self.task.template

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

    def __init__(self, title='', description='', template=''):
        self.title = title
        self.description = description
        self.template = template

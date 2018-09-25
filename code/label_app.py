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

        self.labeller = ModelLabeller(self.model, self.dataset, self.label_helper, logger=logger)

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

    def next_model_labelled_batch(self, size=100):
        model_labelled, target_class = self.dataset.model_labelled
        return model_labelled, target_class

    def next_batch(self, size=10, force_stage=None, reverse_entropy=False, prediction=False):
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

        # Make predictions
        x_to_score = x_data[entropy_indexes]

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


    def add_labels(self, labels):
        is_binary_classification = len(self.label_helper.classes) == 2

        for idx, label in enumerate(labels):
            _id = label['path']
            is_target_class = label['is_target_class']
            save = idx == len(labels) - 1

            if is_target_class:
                self.dataset.add_label(label['path'], label['target_class'], Dataset.TRAIN, user=self.user, save=save)
            else:
                if is_binary_classification:
                    self.dataset.add_label(
                        label['path'],
                        0 if label['target_class'] == 1 else 1,
                        Dataset.TRAIN,
                        user=self.user,
                        save=save,
                    )
                else:
                    # If the task is not binary classification, then its impossible to know what the "other" label is.
                    # Flag this as USER_MODEL_DISAGREEMENT
                    self.dataset.add_label(
                        label['path'],
                        label['target_class'],
                        Dataset.USER_MODEL_DISAGREEMENT,
                        user=self.user,
                        save=save,
                    )

    def add_label(self, _id, label):
        # Validate label
        # TODO: Reevaluate this get_data thing, I'm not a fan of this.
        data = self.dataset.get_data(_id)
        self.label_helper.validate(data, label)
        label = self.label_helper.decode(label)
        # _id is just the path to the file
        self.dataset.add_label(_id, label, self.dataset.current_stage, user=self.user, save=True)

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

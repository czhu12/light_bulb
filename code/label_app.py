from queue import Queue
import numpy as np
import yaml
import time

import pdb

from utils import utils
import logging
import sys
from dataset import Dataset
from training_history import TrainingHistory

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root.addHandler(ch)

class LabelApp:
    @staticmethod
    def load_from(config):
        with open(config) as f:
            config = yaml.load(f)

        task = Task.load_from(config['task'])
        dataset = Dataset.load_from(config['dataset'])
        return LabelApp(task, dataset)

    def __init__(self, task, dataset):
        self.task = task
        self.dataset = dataset
        self.data_type = self.dataset.data_type
        self.model = self.dataset.model
        self.history = TrainingHistory()

    def score(self, request):
        pass

    def next_batch(self, size=10):
        root.debug("Sampling a batch for {} set.".format(self.dataset.current_stage))
        self.dataset.set_current_stage()
        if self.dataset.current_stage == Dataset.TEST:
            sampled_df = self.dataset.sample(size)
            return sampled_df

        # Generate training data
        sampled_df = self.dataset.sample(size * 5)
        if self.data_type == Dataset.IMAGE_TYPE:
            x_data = utils.load_images(sampled_df['path'].values, self.dataset.input_shape)
        if self.data_type == Dataset.TEXT_TYPE:
            x_data = sampled_df['text'].values

        scores = self.model.score(x_data)
        entropy = np.sum(scores * np.log(scores) / np.log(2), axis=1)
        max_entropy_indexes = np.argpartition(-entropy, size)[:size]
        return sampled_df.iloc[max_entropy_indexes]

    def evaluate(self):
        if not self.dataset.ready_to_evaluate:
            raise ValueError(
                "Not ready to test, only has {} samples, needs {} samples.".format(
                    len(self.dataset.test_data),
                    Dataset.MIN_TEST_EXAMPLES,
                )
            )
        x_test, y_test = self.dataset.test_set
        return self.model.evaluate(x_test, y_test)

    def train(self):
        if not self.dataset.ready_to_train:
            raise ValueError(
                "Not ready to train, only has {} samples, needs {} samples.".format(
                    len(self.dataset.train_data),
                    Dataset.MIN_TRAIN_EXAMPLES,
                )
            )
        x_train, y_train = self.dataset.train_set
        root.debug("Training on {} samples.".format(len(x_train)))
        return self.model.train(x_train, y_train)

    def add_label(self, _id, label):
        # _id is just the path to the file
        if label not in Dataset.LABEL_MAP:
            raise ValueError('{} is not a valid label'.format(label))
        label = Dataset.LABEL_MAP[label]
        self.dataset.add_label(_id, label, self.dataset.current_stage)
        root.debug("Unlabelled Count: {}, Labelled Count: {}".format(
            len(self.dataset.unlabelled),
            len(self.dataset.labelled)),
        )
    @property
    def title(self):
        return self.task.title

    def threaded_train(self):
        while True:
            trained = False
            evaluated = False
            if self.history.should_continue_training(len(self.dataset.labelled)):
                try:
                    root.debug("Training model.")
                    history = self.train()

                    root.debug("Training: Loss: {} Acc: {}.".format(
                        history.history['loss'][0],
                        history.history['acc'][0],
                    ))
                    trained = True
                except ValueError as e:
                    print(e)
                try:
                    evaluation = self.evaluate()

                    root.debug("Evaluation: Loss: {} Acc: {}.".format(
                        evaluation[0],
                        evaluation[1],
                    ))
                    evaluated = True
                except ValueError as e:
                    root.error(e)

                if trained and evaluated:
                    self.history.add_train_eval_step(
                        len(self.dataset.labelled), # Length of this dataset might have more labels
                        history.history['acc'][0],
                        history.history['loss'][0],
                        evaluation[1],
                        evaluation[0],
                    )

                    if self.history.should_save_model():
                        root.debug("Saving model.")
                        self.model.save(self.history[-1]['num_samples'])
            else:
                root.debug("Not training model.")

            # Unsupervised training
            try:
                results = self.model.representation_learning(self.dataset.unsupervised_set)
            except ValueError as e:
                root.error(e)

            if len(self.history) > 0:
                print(self.history.plot())

            time.sleep(20)

class Task:
    @staticmethod
    def load_from(config):
        return Task(title=config['title'])

    def __init__(self, title=''):
        self.title = title

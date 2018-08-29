import time
import os
import logging
import threading
from training.training_history import TrainingHistory
import glob
import pickle
from utils.model_evaluation import Evaluator
from dataset import Dataset
from dataset import MIN_UNSUPERVISED_EXAMPLES
from dataset import MIN_TEST_EXAMPLES
from dataset import MIN_TRAIN_EXAMPLES

UNSUP_LEARNING_RATIO = 0.2

THRESHOLD = 0.95
TRAINING_STOPPED = "TRAINING_STOPPED"
TRAINING_STEP_FINISHED = "TRAINING_STEP_FINISHED"

BACKOFF_FACTOR = 1.5
MAX_SLEEP_TIME = 30

class TrainingObserver():
    def notify(self, event, data):
        raise NotImplementedError()

class Trainer():
    def __init__(
            self,
            model_directory,
            model,
            dataset,
            label_helper,
            history=TrainingHistory(),
            logger=logging.getLogger()
        ):
        self.listeners = []
        self.model = model
        self.dataset = dataset
        self.history = history
        self.model_directory = model_directory
        self.label_helper = label_helper
        self.logger = logger

    def notify_listeners(self, event, data):
        for listener in self.listeners:
            listener.notify(event, data)

    def register_listener(self, listener):
        self.listeners.append(listener)

    def get_history(self):
        return self.history

    def load_existing(self, path=None):
        model_paths = glob.glob(os.path.join(self.model_directory, '*'))
        if len(model_paths) == 0:
            return
        performances = [self._parse_performance_from_path(model_path) for model_path in model_paths]
        performances.sort(key=lambda x: self._rank_performance(x), reverse=True)
        perf = performances[0]
        self.logger.debug("Loading weights from {}".format(perf['path']))

        self.model.load_existing_model(perf['model_path'])
        with open(perf['history_path'], 'rb') as f:
            self.history = pickle.load(f)

    def _parse_performance_from_path(self, path):
        basename = path.split('/')[-1]
        _, _, accuracy, _, loss, _, num_labels = basename.split('-')
        return {
            'path': path,
            'model_path': '{}/{}'.format(path, 'model.h5'),
            'history_path': '{}/{}'.format(path, 'history.pkl'),
            'basename': basename,
            'accuracy': accuracy,
            'loss': loss,
            'num_labels': num_labels,
        }

    def _rank_performance(self, performance):
        return performance['num_labels']

    def save(self, performance):
        accuracy = performance['acc']
        loss = performance['loss']
        num_labels = performance['num_labels']

        if not accuracy or not loss or not num_labels:
            raise ValueError("Must have accuracy, loss and num_labels")

        name = 'model-acc-{}-loss-{}-samples-{}'.format(accuracy, loss, num_labels)
        directory = "{}/{}".format(self.model_directory, name)

        self.save_model(directory)
        self.save_training_history(directory)

    # Name will be like: model-accuracy-loss-timestamp
    def save_model(self, directory):
        self.model.save("{}/model.h5".format(directory))

    def save_training_history(self, directory):
        with open("{}/history.pkl".format(directory), 'wb') as output:
            pickle.dump(self.history, output, pickle.HIGHEST_PROTOCOL)

    def ready_to_represent(self, dataset):
        return len(dataset.unlabelled) > MIN_UNSUPERVISED_EXAMPLES

    def ready_to_evaluate(self, dataset):
        return len(dataset.test_data) > MIN_TEST_EXAMPLES

    def ready_to_train(self, dataset):
        return len(dataset.train_data) > MIN_TRAIN_EXAMPLES

    def evaluate(self):
        if not self.ready_to_evaluate(self.dataset):
            raise ValueError(
                "Not ready to test, only has {} samples, needs {} samples.".format(
                    len(self.dataset.test_data),
                    MIN_TEST_EXAMPLES,
                )
            )
        x_test, y_test = self.dataset.test_set
        y_test = self.label_helper.to_training(y_test)
        self.logger.debug("Evaluating on {} samples.".format(len(x_test)))
        return self.model.evaluate(x_test, y_test)

    def train_step(self):
        if not self.ready_to_train(self.dataset):
            raise ValueError(
                "Not ready to train, only has {} samples, needs {} samples.".format(
                    len(self.dataset.train_data),
                    MIN_TRAIN_EXAMPLES,
                )
            )
        x_train, y_train = self.dataset.train_set
        y_train = self.label_helper.to_training(y_train)
        self.logger.debug("Training on {} samples.".format(len(x_train)))
        return self.model.fit(x_train, y_train)

    def print_stats(self):
        stats = self.dataset.stats
        self.logger.debug(stats)

    def train(self):
        sleep_time = 10.
        while True:
            self.print_stats()

            trained = False
            evaluated = False
            if self.history.should_continue_training(len(self.dataset.labelled)):
                just_stopped = False

                # Train
                try:
                    loss, acc = self.train_step()

                    self.logger.debug("Training: Loss: {} Acc: {}.".format(loss, acc))
                    trained = True
                except ValueError as e:
                    sleep_time *= BACKOFF_FACTOR
                    sleep_time = min(sleep_time, MAX_SLEEP_TIME)
                    self.logger.error(e)
                    self.logger.error("Backing off interval time...")

                # Evaluate
                try:
                    evaluation = self.evaluate()

                    self.logger.debug("Evaluation: Loss: {} Acc: {}.".format(
                        evaluation[0],
                        evaluation[1],
                    ))
                    evaluated = True
                except ValueError as e:
                    self.logger.error(e)

                if trained and evaluated:
                    self.history.add_train_eval_step(
                        len(self.dataset.train_data), # Length of this dataset might have more labels
                        loss,
                        acc,
                        evaluation[1],
                        evaluation[0],
                    )

            else:
                self.logger.debug("Not training model.")
                if self.history.should_save_model():
                    self.logger.debug("Saving model.")
                    created_directory = self.save({
                        'num_labels': self.history[-1]['num_labels'],
                        'loss': self.history[-1]['test']['loss'],
                        'acc': self.history[-1]['test']['acc'],
                    })
                    # TODO: Classify more labels

            # Unsupervised training
            try:
                # TODO: If we do this, we have to keep training, because we are changing the underlying model without changing the classifier weights
                if self.ready_to_represent(self.dataset):
                    self.logger.debug("Starting representation training.")
                    x_train, _ = self.dataset.unlabelled_set(MIN_UNSUPERVISED_EXAMPLES)
                    loss = self.model.representation_learning(x_train)
                    self.logger.debug("Finished representation training. Loss: {}".format(loss))
            except ValueError as e:
                self.logger.error(e)

            time.sleep(sleep_time)

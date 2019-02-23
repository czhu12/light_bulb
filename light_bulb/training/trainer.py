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
from saving.model_saver import ModelSaver

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
        self.model = model
        self.history = history
        self.model_saver = ModelSaver(model_directory, model, history)

        self.listeners = []
        self.dataset = dataset
        self.label_helper = label_helper

        self.logger = logger

    def evaluate(self):
        x_test, y_test = self.dataset.test_set
        y_test = self.label_helper.to_training(y_test)
        self.logger.debug("Evaluating on {} samples.".format(len(x_test)))
        return self.model.evaluate(x_test, y_test)

    def train_step(self, verbose=0):
        x_train, y_train = self.dataset.train_set
        y_train = self.label_helper.to_training(y_train)
        self.logger.debug("Training on {} samples.".format(len(x_train)))
        return self.model.fit(x_train, y_train, verbose=verbose)

    def train_epochs(self, epochs=10):
        # This is only used for pretraining
        self.logger.debug("Training for {} epochs".format(epochs))
        for i in range(epochs):
            train_loss, train_acc = self.train_step(verbose=1)
            self.history.add_train_eval_step(
                self.dataset.stats,
                train_acc,
                train_loss,
                0,
                0,
            )

        # Save model
        self.model_saver.save(force=True)

    def load_existing(self):
        # restore history
        history, model = self.model_saver.load()
        self.history = history
        self.model = model

    # Training should happen in two stages:
    # 1. representation pretraining
    # 2. actual training.
    # This information should be saved with the model so training can restart.
    def train(self, total_epochs=None):
        sleep_time = 1.
        # Unsupervised pre-training
        if self.should_learn_to_represent(self.dataset, self.history):
            self.logger.debug("Starting representation training.")
            x_train, _ = self.dataset.unlabelled_set(MIN_UNSUPERVISED_EXAMPLES)
            loss = self.model.representation_learning(x_train)
            self.logger.debug("Finished representation training. Loss: {}".format(loss))
            self.history.finished_representation_learning = True
            self.model_saver.save()

        while True:
            self.print_stats()
            should_keep_training = self.history.should_continue_training(
                    len(self.dataset.labelled))
            ready_to_evaluate = self.ready_to_evaluate(self.dataset)
            ready_to_train = self.ready_to_train(self.dataset)
            if should_keep_training and ready_to_evaluate and ready_to_train:

                train_loss, train_acc = self.train_step()

                self.logger.debug("Training: Loss: {} Acc: {}.".format(train_loss, train_acc))

                eval_loss, eval_acc = self.evaluate()

                self.logger.debug("Evaluation: Loss: {} Acc: {}.".format(
                    eval_loss,
                    eval_acc,
                ))
                self.history.add_train_eval_step(
                    self.dataset.stats,
                    train_acc,
                    train_loss,
                    eval_acc,
                    eval_loss,
                )

            else:
                sleep_time *= BACKOFF_FACTOR
                sleep_time = min(sleep_time, MAX_SLEEP_TIME)
                self.logger.debug("Backing off interval time...")

            if self.history.should_save_model():
                self.logger.debug("Saving model.")
                self.model_saver.save()
                # TODO: Classify more labels

            time.sleep(sleep_time)

    def notify_listeners(self, event, data):
        for listener in self.listeners:
            listener.notify(event, data)

    def register_listener(self, listener):
        self.listeners.append(listener)

    def print_stats(self):
        stats = self.dataset.stats
        self.logger.debug(stats)

    def get_history(self):
        return self.history

    def should_learn_to_represent(self, dataset, history):
        if history.finished_representation_learning:
            return False
        return len(dataset.unlabelled) > MIN_UNSUPERVISED_EXAMPLES

    def ready_to_evaluate(self, dataset):
        return len(dataset.test_data) > MIN_TEST_EXAMPLES

    def ready_to_train(self, dataset):
        return len(dataset.train_data) > MIN_TRAIN_EXAMPLES

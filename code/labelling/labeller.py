import numpy as np
import time
import logging
from utils.model_evaluation import Evaluator
from utils import utils
from labels.label import Label
from dataset import Dataset
from dataset import MIN_TEST_EXAMPLES
from dataset import MIN_TRAIN_EXAMPLES
from dataset import MIN_UNSUPERVISED_EXAMPLES

ACCURACY_RATIO = 1.35 # Only start labeling if we have a 35% accuracy improvement over random classifications
TARGET_PRECISION = 0.94
MAX_INTERVAL_TIME = 60

class ModelLabeller():
    def __init__(
        self,
        model,
        dataset,
        label_helper,
        interval=2,
        logger=logging.getLogger(),
    ):
        self.model = model
        self.dataset = dataset
        self.interval = interval
        self.label_helper = label_helper
        self.logger = logger
        self.exponential_backoff_factor = 0

    def _score_sequence(self, x_test, y_test):
        id2class = self.label_helper.score_classes
        self.logger.debug("Scoring items with model labeller.")
        y_test = self.label_helper.to_training(y_test)
        y_test = utils.one_hot_encode_sequence(
            y_test,
            id2class,
        )
        y_pred = self.model.score(x_test)
        # reshape as a classification problem, to set threshold
        threshold = Evaluator.threshold_for_precision(
            y_test.reshape((y_test.shape[0], -1)),
            y_pred.reshape((y_test.shape[0], -1)),
            TARGET_PRECISION,
        )
        threshold = 0.5

        unlabelled_texts, ids = self.dataset.model_labelling_set()
        if len(unlabelled_texts) == 0:
            self.logger.info("Model labelling done!")
            return 0

        scores = self.model.score(unlabelled_texts)

        dist = scores / scores.sum(axis=-1, keepdims=True)
        idxs = np.argmax(dist, -1)

        num_scored = 0
        self.logger.debug("set labelling threshold as: {}".format(threshold))
        for _id, (text, prediction) in zip(ids, zip(unlabelled_texts, dist)):
            # The prediction has padding so we only take the last len(text) scores.
            text_tag = []
            met_threshold = True
            print("========")
            for word, word_likelihood_dist in zip(text, prediction[-len(text):]):
                print(word_likelihood_dist)
                idx = np.argmax(word_likelihood_dist)
                tag = id2class[idx]
                if np.max(word_likelihood_dist) < threshold:
                    print("Missed threshold")
                    met_threshold = False
                    break
                text_tag.append({'word': word, 'tag': tag})

            if met_threshold:
                print("Met threshold!")
                self.dataset.add_label(
                    _id,
                    self.label_helper.decode(text_tag),
                    stage=Dataset.MODEL_LABELLED,
                    user=Dataset.USER_MODEL_LABELLER,
                    is_labelled=False,
                    save=True,
                )
                num_scored += 1
        return num_scored

    def _score_classification(self, x_test, y_test):
        loss, acc = self.model.evaluate(x_test, y_test)

        num_classes = len(self.label_helper.classes)
        if acc < (1. / num_classes * ACCURACY_RATIO):
            self.logger.debug("Need at least {}% accuracy improvement over naive baseline to start labelling".format(int((ACCURACY_RATIO - 1.) * 100)))
            return 0

        self.logger.debug("Scoring items with model labeller.")
        y_test = utils.one_hot_encode(y_test, num_classes)
        y_pred = self.model.score(x_test)
        threshold = Evaluator.threshold_for_precision(
            y_test,
            y_pred,
            target_precision=TARGET_PRECISION,
        )

        unlabelled, ids = self.dataset.model_labelling_set()
        if len(unlabelled) == 0:
            self.logger.info("Model labelling done!")
            return 0

        scores = self.model.score(unlabelled)
        # if scores is 2 dimentional: (batch x classes)

        # This assumes only classification :(

        # Renormalize scores just in case.
        dist = scores / np.expand_dims(scores.sum(axis=1), -1)
        idxs = np.argmax(dist, -1)

        num_scored = 0
        for _id, (idx, score) in list(zip(ids, zip(idxs, dist[np.arange(len(idxs)), idxs]))):
            if score > threshold:
                num_scored += 1
                self.dataset.add_label(
                    _id,
                    idx,
                    stage=Dataset.MODEL_LABELLED,
                    user=Dataset.USER_MODEL_LABELLER,
                    is_labelled=False,
                    save=True,
                )
        return num_scored

    def start(self):
        while True:
            # Put this at the top of the while loop to prevent thread timing issues.
            # If we want to solve this properly, check here: https://github.com/keras-team/keras/issues/5223
            time.sleep(min(self.exponential_backoff_factor * self.interval, MAX_INTERVAL_TIME))

            x_test, y_test = self.dataset.test_set

            if len(x_test) < MIN_TEST_EXAMPLES:
                self.logger.debug("Need at least {} test labels to start labelling".format(MIN_TEST_EXAMPLES))
                self.exponential_backoff_factor += 1
                continue

            x_train, y_train = self.dataset.train_set

            if len(x_train) < MIN_TRAIN_EXAMPLES:
                self.logger.debug("Need at least {} train labels to start labelling".format(MIN_TRAIN_EXAMPLES))
                self.exponential_backoff_factor += 1
                continue

            # if classification data
            if self.label_helper.label_type == Label.CLASSIFICATION:
                num_scored = self._score_classification(x_test, y_test)
            elif self.label_helper.label_type == Label.SEQUENCE:
                num_scored = self._score_sequence(x_test, y_test)
            else:
                return logger.debug("{} labels aren't support...yet".format(self.label_helper.label_type))

            if num_scored > 0:
                self.exponential_backoff_factor = 0
            else:
                self.exponential_backoff_factor += 1
            self.logger.debug("{} labelled.".format(num_scored))

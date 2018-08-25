import numpy as np
import time
import logging
from utils.model_evaluation import Evaluator
from utils import utils
from dataset import Dataset
from dataset import MIN_TEST_EXAMPLES
from dataset import MIN_TRAIN_EXAMPLES
from dataset import MIN_UNSUPERVISED_EXAMPLES

ACCURACY_RATIO = 1.35 # Only start labeling if we have a 35% accuracy improvement over random classifications
THRESHOLD = 0.98
MAX_INTERVAL_TIME = 60
MODEL_LABELLER = 'MODEL_LABELLER'

class ModelLabeller():
    def __init__(
        self,
        model,
        dataset,
        label_helper,
        interval=10,
        logger=logging.getLogger(),
    ):
        self.model = model
        self.dataset = dataset
        self.interval = interval
        self.label_helper = label_helper
        self.logger = logger
        self.exponential_backoff_factor = 0

    def start(self):
        while True:
            # Put this at the top of the while loop to prevent thread timing issues.
            # If we want to solve this properly, check here: https://github.com/keras-team/keras/issues/5223
            time.sleep(min(self.exponential_backoff_factor * self.interval, MAX_INTERVAL_TIME))

            x_test, y_test = self.dataset.test_set
            num_classes = len(self.label_helper.classes)
            loss, acc = self.model.evaluate(x_test, y_test)
            # TODO(classification_only)

            # TODO(revisit)
            if len(x_test) < MIN_TEST_EXAMPLES:
                self.logger.debug("Need at least {} labels to start labelling".format(MIN_TEST_EXAMPLES))
                self.exponential_backoff_factor += 1
                continue

            if acc < (1. / num_classes * ACCURACY_RATIO):
                self.logger.debug("Need at least {}% accuracy improvement over naive baseline to start labelling".format(int((ACCURACY_RATIO - 1.) * 100)))
                self.exponential_backoff_factor += 1
                continue

            self.logger.debug("Scoring items with model labeller.")
            y_test = utils.one_hot_encode(y_test, num_classes)
            evaluator = Evaluator(self.model, x_test, y_test)
            threshold = evaluator.threshold_for_precision(THRESHOLD)

            unlabelled, ids = self.dataset.unlabelled_set()
            scores = self.model.score(unlabelled)

            # This assumes only classification :(

            # Renormalize scores just in case.
            dist = scores / np.expand_dims(scores.sum(axis=1), -1)
            idxs = np.argmax(dist, -1)

            past_threshold = 0
            for _id, (idx, score) in list(zip(ids, zip(idxs, dist[np.arange(len(idxs)), idxs]))):
                if score > threshold:
                    past_threshold += 1
                    self.dataset.add_label(_id, idx, Dataset.MODEL_LABELLED, MODEL_LABELLER)

            if past_threshold > 0:
                self.exponential_backoff_factor = 0
            else:
                self.exponential_backoff_factor += 1
            self.logger.debug("{} / {} labelled.".format(past_threshold, len(scores)))

import time
import logging
from utils.model_evaluation import Evaluator
from dataset import Dataset
from dataset import MIN_TEST_EXAMPLES
from dataset import MIN_TRAIN_EXAMPLES
from dataset import MIN_UNSUPERVISED_EXAMPLES

THRESHOLD = 0.95
MAX_INTERVAL_TIME = 60

class ModelLabeller():
    def __init__(self, model, dataset, interval=10, logger=logging.getLogger()):
        self.model = model
        self.dataset = dataset
        self.interval = interval
        self.logger = logger
        self.exponential_backoff_factor = 0

    def start(self):
        while True:
            x_test, y_test = self.dataset.test_set
            if len(x_test) > MIN_TEST_EXAMPLES:
                self.logger.debug("Scoring items with model labeller.")
                evaluator = Evaluator(self.model, x_test, y_test)
                threshold = evaluator.threshold_for_precision(THRESHOLD)

                unlabelled, ids = self.dataset.unlabelled_set()
                scores = self.model.score(unlabelled)

                past_threshold = 0
                for id, score in zip(ids, scores):
                    no = score[0]
                    yes = score[1]
                    if no > threshold:
                        past_threshold += 1
                        self.dataset.add_label(id, 0., Dataset.MODEL_LABELLED)
                        self.logger.debug("Labelled {}\tNO.".format(id))

                    if yes > threshold:
                        past_threshold += 1
                        self.dataset.add_label(id, 1., Dataset.MODEL_LABELLED)
                        self.logger.debug("Labelled {}\tYES.".format(id))

                if past_threshold > 0:
                    self.exponential_backoff_factor = 0
                self.logger.debug("{} / {} labelled.".format(past_threshold, len(scores)))
            else:
                self.logger.debug("Need at least {} labels to start labelling images".format(MIN_TEST_EXAMPLES))
                self.exponential_backoff_factor += 1

            time.sleep(min(self.exponential_backoff_factor * self.interval, MAX_INTERVAL_TIME))

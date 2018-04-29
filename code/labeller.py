import time
import logging
from utils.model_evaluation import Evaluator

THRESHOLD = 0.95
MIN_TEST_EXAMPLES = 50

class ModelLabeller():
    def __init__(self, model, dataset, interval=10, logger=logging.getLogger()):
        self.model = model
        self.dataset = dataset
        self.interval = interval
        self.logger = logger

    def start(self):
        while True:
            self.logger.debug("Scoring items with model labeller.")
            x_test, y_test = self.dataset.test_set
            if len(x_test) > MIN_TEST_EXAMPLES:
                evaluator = Evaluator(x_test, y_test)
                threshold = evaluator.threshold_for_precision(THRESHOLD)

                unlabelled, ids = self.dataset.unlabelled_set
                scores = self.model.predict(unlabelled)

                for id, score in zip(ids, scores):
                    no = score[0]
                    yes = score[1]
                    if no > threshold:
                        self.dataset.add_label(id, 'NO')

                    if yes > threshold:
                        self.dataset.add_label(id, 'YES')

            time.sleep(self.interval)

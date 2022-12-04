import numpy as np

class StubModel():
    def representation_learning(self, x_train):
        pass

    def fit(self, x_train, y_train, verbose=0):
        pass

    def evaluate(self, x_train, y_train, verbose=0):
        pass

    # Assume binary classification?
    def score(self, x):
        scores = np.zeros((len(x), 2))
        scores[:, 0] = np.random.rand(len(x))
        scores[:, 1] = 1 - np.random.rand(len(x))
        return scores

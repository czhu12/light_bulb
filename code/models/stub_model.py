import numpy as np
from models.base_model import BaseModel

class StubModel(BaseModel):
    def representation_learning(self, x_train):
        pass

    def fit(self, x_train, y_train):
        pass

    def evaluate(self, x_train, y_train):
        pass

    # Assume binary classification?
    def score(self, x):
        scores = np.zeros((len(x), 2))
        scores[:, 0] = np.random.rand(len(x))
        scores[:, 1] = 1 - np.random.rand(len(x))
        return scores

import pickle
import os

class TFPretrainedModel():
    def __init__(self, directory=None):
        if not directory:
            raise ValueError("Directory must be specified.")
        wrapper_path = os.path.join(directory, 'model.pkl')
        weights_directory = os.path.join(directory, 'weights')
        with open(wrapper_path, "rb") as input_file:
            self.model = pickle.load(input_file)
            self.model.load(weights_directory)

    def train(self, x_train, y_train, validation_split=0., epochs=1):
        return self.model.train(x_train, y_train)

    def score(self, x):
        scores = self.model.score(x_train, y_train)
        assert len(scores.shape) == 3
        return scores

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_train, y_train)

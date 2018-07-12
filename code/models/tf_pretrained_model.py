import importlib
import os

class TFPretrainedModel():
    def __init__(self, abs_klass, directory):
        if not directory:
            raise ValueError("Directory must be specified.")
        wrapper_path = os.path.join(directory, 'model.pkl')
        weights_directory = os.path.join(directory, 'weights')
        module_name = '.'.join(abs_klass.split('.')[:-1])
        klass_name = abs_klass.split('.')[-1]
        print("Loading {} from {}".format(klass_name, module_name))
        module = importlib.import_module(module_name)
        klass = getattr(module, klass_name)
        self.model = klass(directory)

    def train(self, x_train, y_train, validation_split=0., epochs=1):
        return self.model.train(x_train, y_train)

    def score(self, x):
        scores = self.model.score(x)
        return scores

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def representation_learning(self, x_train):
        pass

import os
import pickle

PICKLE_FILE = 'history.pkl'
MODEL_FILE = 'model.h5'

class ModelSaver():
    # Lets only deal with one model for now, which is:
    # best performance for highest test samples so far.
    #
    # The model should be in charge of saving itself, to be able to save
    # things like tag2id and such, but trainer should be injecting the
    # data about performance...

    def __init__(self, directory, model, history):
        self.directory = directory
        self.model = model
        self.history = history

    def save(self, force=False):
        # create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        old_history = self._load_existing_history()
        if self.history.recent_num_samples > old_history.recent_num_samples or force:
            self._save()

    def load(self):
        # Load history
        history = self._load_existing_history()
        # Load model
        model_path = os.path.join(self.directory, MODEL_FILE)
        if os.path.exists(model_path):
            self.model.load_existing_model(model_path)
        return history, self.model

    def _save(self):
        self._save_model()
        self._save_training_history()

    def _load_existing_history(self):
        history_path = os.path.join(self.directory, PICKLE_FILE)

        if not os.path.exists(history_path):
            return self.history

        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        return history

    # Name will be like: model-accuracy-loss-timestamp
    def _save_model(self):
        model_path = os.path.join(self.directory, MODEL_FILE)
        self.model.save(model_path)

    def _save_training_history(self):
        with open(os.path.join(self.directory, PICKLE_FILE), 'wb') as output:
            pickle.dump(self.history, output, pickle.HIGHEST_PROTOCOL)

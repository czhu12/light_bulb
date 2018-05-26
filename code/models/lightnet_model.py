from typing import List
import brambox.boxes as bbb
import lightnet as ln
import utils.lightnet as lightnet_utils

class LightnetModel():
    def __init__(self, classes=[]):
        self.classes = classes
        self.model = lightnet_utils.model_wrapper.YoloModel(classes)

    def load_existing_model(self, path):
        self.model.load_weights(path)

    def save(self, path):
        with self.graph.as_default():
            self.model.save(path)

    def train(self, x_train: List[str], y_train: List[bbb.Box]):
        """
        x_train: List of paths
        y_train: List of bramboxes
        This method will:
        1. Create a torch dataset that will take a list of paths and bramboxes, with appropriate transformations
            Ex: https://gitlab.com/EAVISE/lightnet/blob/master/lightnet/models/_dataset_brambox.py
        2. Create a data loader that will consume the dataset
            Ex: https://gitlab.com/EAVISE/lightnet/blob/master/lightnet/data/_dataloading.py
        3. Train yolo on it by implementing these steps
            Ex: https://gitlab.com/EAVISE/lightnet/blob/master/examples/yolo-voc/train.py#L134-178
                https://gitlab.com/EAVISE/lightnet/blob/master/lightnet/engine/engine.py#L83-115
        """
        dataset = lightnet_utils.brambox_dataset.BramboxDataset(x_train, y_train)

    def representation_learning(self, x_train: List[str], epochs=1):
        """Representation learning makes no sense for this task, does it?"""
        pass

    def score(self, x: List[str]):
        raise NotImplementedError()

    def predict(self, x: List[str]):
        raise NotImplementedError()

    def evaluate(self, x_train: List[str], y_train: List[bbb.Box]):
        raise NotImplementedError()




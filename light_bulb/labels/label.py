import json
#from nltk import word_tokenize
from utils import utils
from utils.text_utils import PAD_TOKEN

class LabelError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(LabelError, self).__init__(message)
        self.message = message

class Label(object):
    CLASSIFICATION = 'classification'
    BINARY = 'binary'
    SEQUENCE = 'sequence'
    OBJECT_DETECTION = 'object_detection'

    @staticmethod
    def load_from(config):
        if 'type' not in config:
            raise ValueError("Please provide a type for label")
        if config['type'] == Label.CLASSIFICATION:
            return ClassificationLabel(config)
        if config['type'] == Label.BINARY:
            return ClassificationLabel({**config, 'classes': ['YES', 'NO']})
        if config['type'] == Label.SEQUENCE:
            return SequenceLabel(**config)
        if config['type'] == Label.OBJECT_DETECTION:
            return ObjectDetectionLabel(**config)
        raise ValueError("Unrecognized label {}".format(config['type']))

    def __init__(self, config):
        self.config = config
        self.label_type = config['type']
        assert self.label_type, "No label type found."

    def validate(self, data, label):
        return True

    def to_training(self, y):
        return y

class ClassificationLabel(Label):
    def __init__(self, config):
        super(ClassificationLabel, self).__init__(config)
        self.classes = config['classes']
        self.label_map = { c: i for i, c in enumerate(self.classes) }

    def decode(self, encoded):
        if encoded not in self.label_map:
            raise LabelError("{} not in {}".format(
                encoded,
                self.label_map,
            ))
        return self.label_map[encoded]

    def to_training(self, y):
        return y
        #return utils.one_hot_encode(y, len(self.classes))

class SequenceLabel(Label):
    def __init__(self, classes=[], default_class=None, **kwargs):
        super(SequenceLabel, self).__init__(kwargs)
        self.default_class = default_class
        self.classes = classes
        self.score_classes = [PAD_TOKEN, self.default_class] + self.classes

    def decode(self, encoded):
        return json.dumps(encoded)

    def to_training(self, ys):
        """y is a list of strings"""
        tags_only = []
        for y in ys:
            tagged_text = json.loads(y)
            tags_only.append([tagged['tag'] for tagged in tagged_text])

        return tags_only

class ObjectDetectionLabel(Label):
    def __init__(self, **kwargs):
        super(ObjectDetectionLabel, self).__init__(kwargs)
        self.classes = kwargs['classes']

    def decode(self, encoded):
        boxes = json.loads(encoded)
        boxes = [{
            'object_id': box['currentClass'],
            'class_label': self.classes[box['currentClass']],
            'x_top_left': box['startX'],
            'y_top_left': box['startY'],
            'width': box['width'],
            'height': box['height'],
        } for box in boxes]

        return json.dumps(boxes)

    def validate(self, boxes):
        boxes = json.loads(boxes)
        assert all(['startX' in box for box in boxes]), "startX not in {}".format(boxes)
        assert all(['startY' in box for box in boxes]), "startY not in {}".format(boxes)
        assert all(['width' in box for box in boxes]), "width not in {}".format(boxes)
        assert all(['height' in box for box in boxes]), "height not in {}".format(boxes)
        assert all(['currentClass' in box for box in boxes]), "currentClass not in {}".format(boxes)

        return True

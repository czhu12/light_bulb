import json
from nltk import word_tokenize
from utils import utils

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
        return utils.one_hot_encode(y, len(self.classes))

class SequenceLabel(Label):
    def __init__(self, length_equality=False, valid_tokens=[], delimiter=' ', **kwargs):
        super(SequenceLabel, self).__init__(kwargs)
        self.length_equality = length_equality
        self.valid_tokens = valid_tokens
        self.delimiter = delimiter

    def validate(self, data, label):
        # check for length_equality
        if self.length_equality:
            label_tokens = label.split(self.delimiter)
            # TODO: This won't work in character-input mode
            if not len(label_tokens) == len(word_tokenize(data)):
                raise LabelError("Label must be of length: {}".format(len(word_tokenize(data))))

        # check valid_tokens are correct
        if self.valid_tokens and len(self.valid_tokens) > 0:
            tokens = label.split(self.delimiter)
            valid_tokens = set(self.valid_tokens)
            if not all(token in valid_tokens for token in tokens):
                raise LabelError("Label {} is invalid. Can only have tokens: {}.".format(label, valid_tokens))

    def decode(self, encoded):
        return encoded

class ObjectDetectionLabel(Label):
    def __init__(self, **kwargs):
        super(ObjectDetectionLabel, self).__init__(kwargs)
        self.classes = kwargs['classes']

    def decode(self, encoded):
        boxes = json.loads(encoded)
        boxes = [{
            'class_label': box['currentClass'],
            'object_id': self.classes.index(box['currentClass']),
            'x_top_left': box['startX'],
            'y_top_left': box['startY'],
            'width': box['width'],
            'height': box['height'],
        } for box in boxes]

        return json.dumps(boxes)

    def validate(self, boxes, label):
        boxes = json.loads(boxes)
        assert all(['startX' in box for box in boxes]), f"startX not in {boxes}"
        assert all(['startY' in box for box in boxes]), f"startY not in {boxes}"
        assert all(['width' in box for box in boxes]), f"width not in {boxes}"
        assert all(['height' in box for box in boxes]), f"height not in {boxes}"
        assert all(['currentClass' in box for box in boxes]), f"currentClass not in {boxes}"

        return True

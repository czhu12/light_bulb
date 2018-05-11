from nltk import word_tokenize

class LabelError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(LabelError, self).__init__(message)
        self.message = message

class Label(object):
    CLASSIFICATION = 'classification'
    BINARY = 'binary'
    SEQUENCE = 'sequence'
    @staticmethod
    def load_from(config):
        if 'type' not in config:
            raise ValueError("Please provide a type for label")
        if config['type'] == Label.CLASSIFICATION:
            return ClassificationLabel(config)
        if config['type'] == Label.BINARY:
            return BinaryClassificationLabel(config)
        if config['type'] == Label.SEQUENCE:
            return SequenceLabel(**config)
        raise ValueError("Unrecognized label {}".format(config['type']))

    def __init__(self, config):
        self.config = config
        self.label_type = config['type']
        assert self.label_type, "No label type found."

    def validate(self, data, label):
        return True

class ClassificationLabel(Label):
    def __init__(self, config):
        super(ClassificationLabel, self).__init__(config)
        self.classes = config['classes']

    def decode(self, encoded):
        return encoded

class BinaryClassificationLabel(ClassificationLabel):
    LABEL_MAP = { 'YES': 1, 'NO': 0 }
    def __init__(self, config):
        super(BinaryClassificationLabel, self).__init__({**config, **{'classes': ['YES', 'NO']}})

    def decode(self, encoded):
        if encoded not in BinaryClassificationLabel.LABEL_MAP:
            raise LabelError("{} not in {}".format(
                encoded,
                BinaryClassificationLabel.LABEL_MAP,
            ))
        return encoded

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

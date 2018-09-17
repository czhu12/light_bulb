from cerberus import Validator
import yaml
import os

class ConfigParser:
    schema = {
        'task': {
            'type': 'dict',
            'schema': {
                'title': {'type': 'string', 'required': True},
                'template': {'type': 'string', 'required': False},
            },
            'required': True,
        },
        'dataset': {
            'type': 'dict',
            'schema': {
                'directory': {'type': 'string', 'required': True},
                'data_type': {'type': 'string', 'required': True, 'allowed': ['images', 'text', 'json']},
                'judgements_file': {'type': 'string', 'required': True},
            },
            'required': True,
        },
        'model': {
            'type': 'dict',
            'schema': {
                'directory': {'type': 'string', 'required': True},
            },
            'required': True,
        },
        'label': {
            'type': 'dict',
            'schema': {
                'type': {'type': 'string', 'required': True, 'allowed': ['classification', 'binary', 'sequence', 'object_detection']},
                'classes': {'type': 'list', 'required': False},
                'default_class': {'type': 'string', 'required': False},
            },
            'required': True,
        },
        'user': {
            'type': 'string',
            'required': True,
        },
    }

    def __init__(self, config):
        self.config = config
        self.validator = Validator(ConfigParser.schema)
        if len(self.validator.errors) > 0:
            raise ValueError("Error in config: {}".format(self.validator.errors))
        self.task = self.config['task']
        self.dataset = self.config['dataset']
        self.model = self.config['model']
        self.label = self.config['label']
        self.user = self.config['user']

    def _create_directories(self):
        """Create model directory and judgements directory."""
        directory = os.path.dirname(self.config['dataset']['judgements_file'])
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = self.config['model']['directory']
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def load(path):
        with open(path) as f:
            config = yaml.load(f)

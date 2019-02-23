import pdb
import os
import plac
import yaml
import numpy as np
import logging
from flask import Flask, request, send_from_directory, render_template, jsonify, send_file
from label_app import LabelApp
from labels.label import LabelError
from mimetypes import MimeTypes
from threading import Thread
from utils import utils
from dataset import MIN_TRAIN_EXAMPLES
from dataset import MIN_TEST_EXAMPLES
from dataset import MIN_UNSUPERVISED_EXAMPLES

app = Flask(__name__, static_folder='ui/build/static', template_folder='ui/build')

class Server():
    """This wraps a label app and serves json information about it."""
    def __init__(self, label_app):
        self.label_app = label_app

    def get_task(self):
        valid_tokens = []
        if hasattr(self.label_app.label_helper, 'valid_tokens'):
            valid_tokens = self.label_app.label_helper.valid_tokens

        classes = None
        if hasattr(self.label_app.label_helper, 'classes'):
            classes = self.label_app.label_helper.classes

        default_class = None
        if hasattr(self.label_app.label_helper, 'default_class'):
            default_class = self.label_app.label_helper.default_class

        return jsonify({
            'title': self.label_app.title,
            'description': self.label_app.description,
            'template': self.label_app.template,
            'label_type': self.label_app.label_helper.label_type,
            'data_type': self.label_app.data_type,
            'classes': classes,
            'default_class': default_class,
            'valid_tokens': valid_tokens,
            "min_train": MIN_TRAIN_EXAMPLES,
            "min_test": MIN_TEST_EXAMPLES,
            "min_unsup": MIN_UNSUPERVISED_EXAMPLES,
            "config": self.label_app.config,
        })

    def create_judgement(self):
        try:
            data = request.get_json()
            id = data.get('id')
            label = data.get('label')
            time_taken = data.get('time_taken')
            self.label_app.add_label(id, label, time_taken=time_taken)
            return jsonify({'id': id, 'label': label})
        except LabelError as e:
            return jsonify({'error': e.message})

    def create_judgements(self):
        try:
            data = request.get_json()
            labels = data.get('labels')
            avg_time_taken = data.get('time_taken')
            self.label_app.add_labels(labels, avg_time_taken=avg_time_taken)
            return jsonify(data)
        except LabelError as e:
            return jsonify({'error': e.message})

    def labeled_data(self):
        """
        Returns labeled rows from the dataset.
        """
        page = int(request.args.get('page'))
        page_size = int(request.args.get('page_size'))
        labelled = True if request.args.get('labelled') == 'true' else False

        batch, done = self.label_app.labelled_data(page, (page + 1) * page_size, labelled)
        batch = batch.fillna('NaN')
        return jsonify({
            "dataset": list(batch.T.to_dict().values()),
            "done": done,
        })

    def get_batch(self):
        """
        Returns a batch of items to be labelled and the phase
        of the label collection (training or test)
        """
        if self.label_app.is_done:
            return jsonify({
                "done": True,
                "batch": [],
                "entropy": [],
                "stage": [],
                "y_prediction": [],
            })

        prediction = request.args.get('prediction') and request.args.get('prediction') == 'true'
        reverse_entropy = request.args.get('reverse_entropy') and request.args.get('reverse_entropy') == 'true'

        kwargs = {'prediction': prediction}
        if request.args.get('force_stage'):
            kwargs['force_stage'] = request.args.get('force_stage')

        if request.args.get('sample_size'):
            kwargs['size'] = int(request.args.get('sample_size'))
        kwargs['reverse_entropy'] = reverse_entropy

        batch, stage, y_prediction, entropies = self.label_app.next_batch(**kwargs)

        batch = batch.fillna('NaN')
        json_batch = jsonify({
            "batch": list(batch.T.to_dict().values()),
            "entropy": entropies,
            "stage": stage,
            "y_prediction": y_prediction,
            "done": False,
        })

        return json_batch

    def batch_items_batch(self):
        # TODO: Fix this shit.
        search_query = request.args.get('search_query')
        target_size = 10 if self.label_app.label_helper.label_type == 'sequence' else 100
        if search_query:
            # if search query is present, we need to call label_app.search
            batch = self.label_app.search(search_query)
            target_class = -1
        else:
            batch, target_class = self.label_app.next_model_labelled_batch(target_size)

        if len(batch) == 0:
            return jsonify({
                "batch": [],
                "target_class": 0,
                "done": True,
            })
        else:
            batch = batch.fillna('NaN')
            json_batch = jsonify({
                "batch": list(batch.T.to_dict().values()),
                "target_class": target_class,
                "done": False,
            })

            return json_batch

    def get_history(self):
        """
        Get training history for label app.
        """
        history = {
            "history": self.label_app.get_history().history,
        }
        stats = self.label_app.get_stats()
        return jsonify({
            **history,
            **stats,
        })

    def score(self):
        """
        Image Classification:
        {
            "type": "images",
            "urls": ["https://my-url.com/image.jpg"],
        }
        Text Classification:
        {
            "type": "texts",
            "texts": ["the text that i want to classify"],
        }
        Sequence Classification:
        {
            "type": "sequence",
            "texts": ["the sequence that i want to label"],
        }
        """
        json = request.get_json()

        _type = json['type']
        if _type == "images":
            urls = json["urls"]
            x_train, images = utils.download_urls(urls)
            scores = self.label_app.score(x_train)
        elif _type == "text" or _type == "sequence":
            texts = json["texts"]
            scores = self.label_app.score(texts)
        return jsonify({'scores': scores.tolist(), 'labels': self.label_app.label_helper.classes})

    def predict(self):
        """
        Image Classification:
        {
            "type": "images",
            "urls": ["https://my-url.com/image.jpg"],
        }
        Text Classification:
        {
            "type": "texts",
            "texts": ["the text that i want to classify"],
        }
        Sequence Classification:
        {
            "type": "sequence",
            "texts": ["the sequence that i want to label"],
        }
        """
        json = request.get_json()

        _type = json['type']
        if _type == "images":
            urls = json["urls"]
            x_train, images = utils.download_urls(urls)
            predictions = self.label_app.predict(x_train)
        elif _type == "text" or _type == "sequence":
            texts = json["texts"]
            predictions = self.label_app.predict(texts)
        return jsonify({'predictions': predictions.tolist(), **json})

@app.route('/')
@app.route('/dataset')
def index():
    return render_template(
        'index.html',
        data_type=server.label_app.data_type,
        title=server.label_app.title,
        description=server.label_app.description,
        label_helper=server.label_app.label_helper,
        label_type=server.label_app.label_helper.label_type,
    )

@app.route('/api/task', methods=['GET'])
def task():
    """
    Returns:
        task: {
            title,
            description,
        }
    """
    return server.get_task()

@app.route('/api/judgements', methods=['POST'])
def create_judgement():
    return server.create_judgement()

@app.route('/api/judgements/batch', methods=['POST'])
def create_judgements():
    return server.create_judgements()

@app.route('/api/dataset', methods=['GET'])
def labeled_data():
    return server.labeled_data()

@app.route('/api/batch', methods=['GET'])
def batch():
    return server.get_batch()


def _df_to_jsonable(df):
    rows = df.T.to_dict().values()
    """This is a really annoying hack, but we need to convert all np.int's to ints"""
    for row in rows:
        for key, value, in row.items():
            if type(value) == np.int or type(value) == np.int32 or type(value) == np.int64:
                row[key] = int(value)
            if type(value) == np.float or type(value) == np.float32 or type(value) == np.float64:
                row[key] = float(value)

    return rows


@app.route('/api/batch_items_batch', methods=['GET'])
def batch_items_batch():
    # TODO: Fix this shit.
    return server.batch_items_batch()

@app.route('/api/history', methods=['GET'])
def history():
    return server.get_history()

@app.route('/api/evalution', methods=['GET'])
def evaluation():
    """
    Evalaute current model
    """
    return jsonify(label_app.evaluate())

@app.route('/images')
def get_image():
    image_path = request.args.get('image_path')
    mime = MimeTypes()
    mimetype, _ = mime.guess_type(image_path)
    return send_file(os.path.join('..', image_path), mimetype=mimetype)

@app.route("/demo")
def demo():
    return render_template(
        'index.html',
        data_type=server.label_app.data_type,
        title=server.label_app.title,
        description=server.label_app.description,
        label_helper=server.label_app.label_helper,
        label_type=server.label_app.label_helper.label_type,
    )

@app.route("/api/score", methods=['POST', 'PUT', 'GET'])
def score():
    return server.score()

@app.route("/api/predict", methods=['POST', 'PUT', 'GET'])
def predict():
    return server.predict()

@app.route('/css/index.css')
def root():
    return send_from_directory('ui/src', 'index.css')

@plac.annotations(
    config=("Path to config file", "option", "c", str),
    port=("Port to start server", "option", "p", int),
    mode=("Production or training mode", "option", "m", str, ["production", "training"]),
    log_level=("Log level.", "option", "l", str, ['DEBUG', 'INFO', 'ERROR']),
    no_train=("Don't train model.", "flag", "d", bool),
)
def main(config, port=5000, mode="training", log_level='DEBUG', no_train=False):
    global server
    label_app = LabelApp.load_from({'path': config, 'log_level': log_level})
    server = Server(label_app)

    if not no_train:
        train_thread = Thread(target=label_app.threaded_train)
        train_thread.daemon = True
        train_thread.start()

    if not no_train:
        labelling_thread = Thread(target=label_app.threaded_label)
        labelling_thread.daemon = True
        labelling_thread.start()

    print("Setting log level to {}".format(log_level))
    print("Started local server at http://localhost:5000")
    app.logger.setLevel(getattr(logging, log_level))
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=port)

if __name__ == "__main__":
    plac.call(main)

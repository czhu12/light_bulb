import pdb
import plac
import yaml
from flask import Flask, request, send_from_directory, render_template, jsonify, send_file
from label_app import LabelApp
from label_app import CLASSIFICATION_COLORS
from label import LabelError
from mimetypes import MimeTypes
from threading import Thread
from utils import utils

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template(
        'index.html',
        data_type=label_app.data_type,
        title=label_app.title,
        description=label_app.description,
        label_helper=label_app.label_helper,
        label_type=label_app.label_type,
        classification_colors=CLASSIFICATION_COLORS,
    )

@app.route('/judgements', methods=['POST'])
def create_judgement():
    try:
        id = request.form.get('id')
        label = request.form.get('label')
        label_app.add_label(id, label)
        return jsonify({'id': id, 'label': label})
    except LabelError as e:
        return jsonify({'error': e.message})

@app.route('/batch', methods=['GET'])
def batch():
    """
    Returns a batch of items to be labelled and the phase
    of the label collection (training or test)
    """
    if label_app.is_done:
        return jsonify({
            "done": True,
            "batch": [],
            "entropy": [],
            "stage": [],
            "y_prediction": [],
        })

    prediction = request.args.get('prediction') and request.args.get('prediction') == 'true'
    batch, indexes, stage, x_data = label_app.next_batch()
    y_prediction = None
    if prediction and x_data and len(x_data) > 0:
        y_prediction = label_app.predict(x_data)

    batch = batch.fillna('NaN')
    json_batch = jsonify({
        "batch": list(batch.T.to_dict().values()),
        "entropy": indexes,
        "stage": stage,
        "y_prediction": y_prediction,
        "done": False,
    })

    return json_batch

@app.route('/history', methods=['GET'])
def history():
    """
    Get training history for label app.
    """
    history = {
        "history": label_app.get_history().history,
    }
    stats = label_app.get_stats()
    return jsonify({
        **history,
        **stats,
    })

@app.route('/evalution', methods=['GET'])
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
    return send_file(image_path, mimetype=mimetype)

@app.route("/demo")
def demo():
    return render_template(
        'demo.html',
        data_type=config_obj['dataset']['data_type'],
        title=label_app.title,
    )

@app.route("/score", methods=['POST', 'PUT', 'GET'])
def score():
    """
    Image Classification:
    {
        "type": "image",
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

    type = json['type']
    if type == "images":
        urls = json["urls"]
        x_train, images = utils.download_urls(urls)
        scores = label_app.score(x_train)
    elif type == "text":
        texts = json["texts"]
        scores = label_app.score(texts)
    return jsonify({'scores': scores.tolist()})

@app.route("/predict", methods=['POST', 'PUT', 'GET'])
def predict():
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
        predictions = label_app.predict(x_train)
    elif _type == "text" or _type == "sequence":
        texts = json["texts"]
        predictions = label_app.predict(texts)
    return jsonify({'predictions': predictions})

@plac.annotations(
    config=("Path to config file", "option", "c", str),
    port=("Port to start server", "option", "p", int),
    mode=("Production or training mode", "option", "m", str, ["production", "training"])
)
def main(config, port=5000, mode="training"):
    global label_app
    global config_obj
    config_obj = yaml.load(open(config, 'r'))

    label_app = LabelApp.load_from(config)

    train_thread = Thread(target=label_app.threaded_train)
    train_thread.daemon = True
    train_thread.start()

    #labelling_thread = Thread(target=label_app.threaded_label)
    #labelling_thread.daemon = True
    #labelling_thread.start()

    print("Started local server at http://localhost:5000")
    app.debug = True
    app.run(debug=True, port=port)

if __name__ == "__main__":
    plac.call(main)

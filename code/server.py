import pdb
import plac
import yaml
from flask import Flask, request, send_from_directory, render_template, jsonify, send_file
from label_app import LabelApp
from mimetypes import MimeTypes
from threading import Thread
from utils import utils

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template(
        'index.html',
        data_type=config_obj['dataset']['data_type'],
        title=label_app.title,
    )

@app.route('/judgements', methods=['POST'])
def create_judgement():
    id = request.form.get('id')
    label = request.form.get('label')
    label_app.add_label(id, label)
    return jsonify({'id': id, 'label': label})

@app.route('/batch', methods=['GET'])
def batch():
    """
    Returns a batch of items to be labelled and the phase
    of the label collection (training or test)
    """
    batch = label_app.next_batch()
    batch = batch.fillna('NaN')
    json_batch = jsonify({'batch': list(batch.T.to_dict().values())})
    return json_batch

@app.route('/history', methods=['GET'])
def history():
    """
    Get training history for label app.
    """
    return jsonify({ "history": label_app.get_history().history })

@app.route('/evalution', methods=['GET'])
def evaluation():
    """
    Evalaute current model
    """
    return jsonify(label_app.evaluate())

@app.route('/images')
def get_image():
    image_path = request.args.get('image_path')
    print(image_path)
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
        "type": "image_classification",
        "urls": ["https://my-url.com/image.jpg"],
    }
    Text Classification:
    {
        "type": "text_classification",
        "texts": ["the text that i want to classify"],
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

@plac.annotations(
    config=("Path to config file", "option", "c", str),
    mode=("Production or training mode", "option", "m", str, ["production", "training"])
)
def main(config, mode="training"):
    global label_app
    global config_obj
    config_obj = yaml.load(open(config, 'r'))

    label_app = LabelApp.load_from(config)

    train_thread = Thread(target=label_app.threaded_train)
    train_thread.daemon = True
    train_thread.start()

    labelling_thread = Thread(target=label_app.threaded_label)
    labelling_thread.daemon = True
    labelling_thread.start()

    app.debug = True
    app.run(debug=True)

if __name__ == "__main__":
    plac.call(main)

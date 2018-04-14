import pdb
import plac
import yaml
from flask import Flask, request, send_from_directory, render_template, jsonify, send_file
from label_app import LabelApp
from mimetypes import MimeTypes
from threading import Thread

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

@app.route("/score")
def score():
    """
    Image Classification:
    {
        "url": "https://my-url.com/image.jpg",
    }
    Text Classification:
    {
        "text": "the text that i want to classify",
    }
    """
    self.label_app.score(request)

@plac.annotations(
    config=("Path to config file", "option", "c", str),
    mode=("Production or training mode", "option", "m", str, ["production", "training"])
)
def main(config, mode="training"):
    global label_app
    global config_obj
    config_obj = yaml.load(open(config, 'r'))

    label_app = LabelApp.load_from(config)

    t = Thread(target=label_app.threaded_train)
    t.daemon = True
    t.start()

    app.run(debug=True)

if __name__ == "__main__":
    plac.call(main)

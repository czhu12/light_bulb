import numpy as np
import keras
from keras.preprocessing import image as image_utils
from keras.applications import imagenet_utils
from keras.preprocessing import sequence
from collections import Counter
from nltk import word_tokenize
from PIL import Image
import requests
from io import BytesIO
import os
import logging

UNKNOWN_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

logger = logging.getLogger()


def unfreeze_layers(net):
    net.trainable = True
    for l in net.layers:
        l.trainable = True

def freeze_layers(net):
    net.trainable = False
    for l in net.layers:
        l.trainable = False

def load_images(image_paths, input_shape):
    images = np.zeros((len(image_paths), input_shape[0], input_shape[1], 3))

    for (idx, image_path) in enumerate(image_paths):
        images[idx] = image_utils.load_img(image_path, target_size=input_shape)

    return imagenet_utils.preprocess_input(images)

def one_hot_encode(y, num_classes):
    return keras.utils.to_categorical(y, num_classes=num_classes)

def download_urls(urls, target_size=(128, 128)):
    images = []
    for url in urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img = img.resize(target_size)
        images.append(img)

    return np.array([np.array(img) for img in images]), images

def download_file(remote_path, local_dir):
    if not os.path.isdir('./vendor'):
        logger.debug("Created ./vendor")
        os.makedirs('./vendor')

    if not os.path.isdir(local_dir):
        logger.debug(f"Created {local_dir}")
        os.makedirs(local_dir)

    filename = os.path.basename(remote_path)
    path_to_downloaded_file = f'{local_dir}/{filename}'
    if os.path.exists(path_to_downloaded_file):
        return path_to_downloaded_file

    with open(path_to_downloaded_file, "wb") as f:
        response = requests.get(remote_path)
        f.write(response.content)

    return path_to_downloaded_file

# Token wise one hot encode.
def one_hot_encode_sequence(y_seqs, id2class):
    id2class = [PAD_TOKEN, EOS_TOKEN] + list(id2class)
    class2id = { c: i for i, c in enumerate(id2class) }
    for y_seq in y_seqs:
        assert all(y in id2class for y in y_seq), "{} not in valid_tokens: {}".format(set(y_seq) - set(id2class), id2class)

    _to_seq_ids = lambda seq: [class2id[y] for y in seq]
    y_seq_ids = [_to_seq_ids(y_seq) for y_seq in y_seqs]
    ys = sequence.pad_sequences(y_seq_ids, value=class2id[PAD_TOKEN])
    y_one_hot = np.zeros((len(ys), len(ys[0]), len(class2id)))
    for idx, y in enumerate(ys):
        for idx2, y2 in enumerate(y):
            y_one_hot[idx][idx2][y2] = 1.

    return y_one_hot

def decode_one_hot_sequence_predictions(y_scores, lengths, id2class):
    id2class = [PAD_TOKEN, EOS_TOKEN] + list(id2class)
    class2id = { t: i for i, t in enumerate(id2class) }
    decoded = []
    for index, y_score in enumerate(y_scores):
        tags = []
        idxs = np.argmax(y_score, axis=1)
        for idx in idxs:
            tags.append(id2class[idx])

        length = lengths[index]
        decoded.append(tags[-length:-1])
    # Filter out all preset tokens
    return decoded


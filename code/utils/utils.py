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

def one_hot_encode(y):
    return keras.utils.to_categorical(y)

def download_urls(urls, target_size=(128, 128)):
    images = []
    for url in urls:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize(target_size)
        images.append(img)

    return np.array([np.array(img) for img in images]), images
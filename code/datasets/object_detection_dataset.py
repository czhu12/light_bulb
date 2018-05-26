import os
import copy
import logging
import json
from PIL import Image
import brambox.boxes as bbb
import lightnet.data as lnd
from dataset import ImageDataset


def serialize_brambox(box) -> str:
    return json.dumps({
        'class_label': box.class_label,
        'object_id': box.object_id,
        'x_top_left': box.x_top_left,
        'y_top_left': box.y_top_left,
        'width': box.width,
        'height': box.height,
    })

def deserialize_brambox(encoding: str) -> bbb.Box:
    obj = json.loads(encoding)
    box = bbb.Box()
    box.class_label = obj['class_label']
    box.object_id = obj['object_id']
    box.x_top_left = obj['x_top_left']
    box.y_top_left = obj['y_top_left']
    box.width = obj['width']
    box.height = obj['height']
    return box

class ObjectDetectionDataset(ImageDataset):
    def test_set(self):
        if len(self.test_data) == 0:
            return [], []
        test_data = self.test_data
        paths = train_data['path'].values
        boxes = [deserialize_brambox(label) for label in train_data['label'].values]

    def train_set(self):
        if len(self.train_data) == 0:
            return [], []
        train_data = self.train_data
        paths = train_data['path'].values
        boxes = [deserialize_brambox(label) for label in train_data['label'].values]

"""This dataset represents a single training or test set"""
from typing import List, Tuple
import brambox.boxes as bbb
import lightnet.data as lnd
from torchvision import transforms as torch_transforms

DEFAULT_SIZE = (416, 416)
JITTER = 0.2
FLIP = 0.5
HUE = 0.1
SAT = 1.5
VAL = 1.5

class BramboxDataset(lnd.Dataset):
    @staticmethod
    def build_default(image_paths: List[str], annotations: List[bbb.Box], input_dimension=DEFAULT_SIZE):
        lb = lnd.transform.Letterbox(dimension=input_dimension)
        rf = lnd.transform.RandomFlip(FLIP)
        rc = lnd.transform.RandomCrop(JITTER, True, 0.1)
        hsv = lnd.transform.HSVShift(HUE, SAT, VAL)
        it = torch_transforms.ToTensor()
        image_transforms = lnd.transform.Compose([hsv, rc, rf, lb, it])
        annotation_transforms = lnd.transform.Compose([rc, rf, lb])
        return BramboxDataset(image_paths, annotations, input_dimension, image_transforms, annotation_transforms)

    def __init__(self, image_paths: List[str], annotations: List[bbb.Box], input_dimension=DEFAULT_SIZE, image_transforms=None, annotation_transforms=None):
        """
        Args:
            image_paths: list of image paths
            annotations (List): (bbb.Box) list of brambox bounding boxes
            input_dimension (tuple): (width,height) tuple with default dimensions of the network
            img_transform (torchvision.transforms.Compose): Transforms to perform on the images
            annotation_transforms (torchvision.transforms.Compose): Transforms to perform on the annotations
        """
        super().__init__(input_dimension)
        assert len(image_paths) == len(annotations)
        self.image_transforms = image_transforms
        self.annotation_transforms = annotation_transforms
        self.image_paths = image_paths
        self.annotations = annotations

    def __len__(self):
        return len(self.image_paths)

    @lnd.Dataset.resize_getitem
    def __getitem__(self, index):
        """ Get transformed image and annotations based of the index of ``self.keys``

        Args:
            index (int): index of the `image_paths`, `annotations` list containing all the image identifiers of the dataset.

        Returns:
            tuple: (transformed image, list of transformed brambox boxes)
        """
        if index >= len(self):
            raise IndexError(f'list index out of range [{index}/{len(self)-1}]')

        # Load
        img = Image.open(self.image_paths[index])
        anno = copy.deepcopy(self.annos[index])

        # Transform
        if self.image_transforms is not None:
            img = self.image_transforms(img)
        if self.annotation_transforms is not None:
            annotation = self.annotation_transforms(annotation)

        return img, annotation

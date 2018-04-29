from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import pdb

class Evaluator():
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
    
    def threshold_for_precision(self, target_precision=0.9):
        y_score = self.model.score(self.X)
        average_precision = average_precision_score(self.y, y_score)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
        index = np.argmax(precisions > target_precision)
        threshold = thresholds[index]
        return threshold

from sklearn.metrics import precision_recall_curve
import numpy as np

class Evaluator():
    @staticmethod
    def threshold_for_precision(y_true, y_pred, target_precision):
        # TODO: This is only binary classification. WTF?
        if len(y_true.shape) == 2:
            return Evaluator._threshold_for_classification(y_true, y_pred, target_precision)
        elif len(y_true.shape) == 3:
            return Evaluator._threshold_for_sequence(y_true, y_pred, target_precision)

    @staticmethod
    def _threshold_for_classification(y_true, y_pred, target_precision):
        target_thresholds = []
        for i in range(y_true.shape[-1]):
            precisions, recalls, thresholds = precision_recall_curve(y_true[:, 0], y_pred[:, 0])
            index = np.argmax(precisions > target_precision)
            threshold = thresholds[index - 1]
            target_thresholds.append(threshold)

        # Take the average threshold
        return sum(target_thresholds) / len(target_thresholds)

    @staticmethod
    def _threshold_for_sequence(y_true, y_pred, target_precision):
        # TODO: Finish this properly
        pass

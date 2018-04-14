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
        y_score = classifier.decision_function(self.X)
        average_precision = average_precision_score(self.y, y_score)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_score)
        index = np.argmax(precisions > target_precision)
        threshold = thresholds[index]
        return threshold

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Add noisy features
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # Limit to the two first classes, and split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
                                                        test_size=.5,
                                                        random_state=random_state)

    # Create a simple classifier
    classifier = svm.LinearSVC(random_state=random_state)
    classifier.fit(X_train, y_train)
    # Create evaluator object and get threshold for 90% precision
    evaluator = Evaluator(classifier, X_test, y_test)
    print(evaluator.threshold_for_precision(0.9))

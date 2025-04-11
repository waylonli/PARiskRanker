import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone, BaseEstimator, ClassifierMixin
import importlib

class ThresholdClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, refit=True, val_size=0.3):
        self.estimator = estimator
        self.refit = refit
        self.val_size = val_size

    def fit(self, X, y):

        def scoring(th, y, prob):
            pred = (prob > th).astype(int)
            return 0 if not pred.any() else \
                -fbeta_score(y, pred, beta=0.1)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, stratify=y, test_size=self.val_size,
            shuffle=True, random_state=1234
        )

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_train, y_train)

        prob_val = self.estimator_.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0, 1, 200)[1:-1]
        scores = [scoring(th, y_val, prob_val)
                  for th in thresholds]
        self.score_ = np.min(scores)
        self.th_ = thresholds[np.argmin(scores)]

        if self.refit:
            self.estimator_.fit(X, y)
        if hasattr(self.estimator_, 'classes_'):
            self.classes_ = self.estimator_.classes_

        return self

    def predict(self, X):
        proba = self.estimator_.predict_proba(X)[:, 1]
        return (proba > self.th_).astype(int)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)



def instantiate_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_()
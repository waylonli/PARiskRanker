import numpy as np
from sklearn.metrics import fbeta_score, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone, BaseEstimator, ClassifierMixin
import importlib
from tqdm import tqdm


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


def compute_ranking_metrics(df):
    ndcg_3 = 0
    ndcg_5 = 0
    ndcg_10 = 0
    mrr = 0
    validate_qids = 0

    for qid in tqdm(df['qid'].unique(), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                    desc="Computing ranking metrics"):
        group = df[df['qid'] == qid][['fst_step_scores', 'Class']]
        if len(group) <= 1:
            continue
        target_label = group['Class'].values.astype(int)
        if max(target_label) == 0 or len(target_label) < 2:
            continue

        rank_score = group['fst_step_scores'].values

        if max(rank_score) > 1 or min(rank_score) < 0:
            rank_score = 1 / (1 + np.exp(-rank_score.astype(np.float64)))
        # Sort by predicted score descending for correct MRR calculation
        order = np.argsort(-rank_score)
        labels_sorted = target_label[order]
        scores_sorted = rank_score[order]
        # NDCG: use labels and scores in sorted order for consistency
        ndcg_3 += ndcg_score([labels_sorted], [scores_sorted], k=3)
        ndcg_5 += ndcg_score([labels_sorted], [scores_sorted], k=5)
        ndcg_10 += ndcg_score([labels_sorted], [scores_sorted], k=10)

        # MRR: reciprocal of rank of first relevant (Class 1)
        if 1 in labels_sorted:
            rank = np.where(labels_sorted == 1)[0][0] + 1
            mrr += 1 / rank

        validate_qids += 1

    if validate_qids == 0:
        return 0, 0, 0, 0
    ndcg_3 /= validate_qids
    ndcg_5 /= validate_qids
    ndcg_10 /= validate_qids
    mrr /= validate_qids

    return ndcg_3, ndcg_5, ndcg_10, mrr

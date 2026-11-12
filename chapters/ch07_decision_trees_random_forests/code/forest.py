# ci: skip
"""Random forest from scratch: bagging of CART trees with random feature subsets and an out-of-bag estimate."""
import numpy as np

from tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=50, criterion="gini", max_depth=None, max_features="sqrt", bootstrap=True, seed=0):
        self.n_trees, self.criterion, self.max_depth = n_trees, criterion, max_depth
        self.max_features, self.bootstrap, self.seed = max_features, bootstrap, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y).astype(int)
        rng = np.random.default_rng(self.seed)
        n, self.k_ = len(y), int(y.max()) + 1
        self.trees_, self.oob_ = [], []
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, n) if self.bootstrap else np.arange(n)
            t = DecisionTree(self.criterion, self.max_depth, max_features=self.max_features,
                             rng=np.random.default_rng(rng.integers(1 << 31)))
            t.fit(X[idx], y[idx])
            self.trees_.append(t)
            self.oob_.append(np.setdiff1d(np.arange(n), idx))
        self.n_features_ = X.shape[1]
        self.X_, self.y_ = X, y
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)   # soft voting

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def oob_score(self):
        """Each sample is classified only by the trees that did not see it."""
        votes = np.zeros((len(self.y_), self.k_))
        for t, oob in zip(self.trees_, self.oob_):
            if len(oob):
                votes[oob] += t.predict_proba(self.X_[oob])
        seen = votes.sum(axis=1) > 0
        return float((votes[seen].argmax(axis=1) == self.y_[seen]).mean())

    def feature_importances(self):
        return np.mean([t.feature_importances() for t in self.trees_], axis=0)

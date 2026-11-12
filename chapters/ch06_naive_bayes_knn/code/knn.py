# ci: run
"""k-nearest neighbours from scratch: distances, classification, regression, standardisation."""
import numpy as np


def minkowski(a, b, p=2.0):
    """Minkowski distance (sum |a_j - b_j|^p)^(1/p); p=1 Manhattan, p=2 Euclidean, p=inf Chebyshev."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return float(d.max()) if np.isinf(p) else float((d ** p).sum() ** (1.0 / p))


def standardise(X_train, X_test):
    """Scale with the mean and standard deviation of the TRAINING data only."""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


class KNN:
    """Lazy learner: fit() only stores the data. Ties in a vote go to the nearest of the tied classes."""

    def __init__(self, k=3, p=2.0, weights="uniform", task="classification"):
        self.k, self.p, self.weights, self.task = k, p, weights, task

    def fit(self, X, y):
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y)
        return self

    def _distances(self, x):
        d = np.abs(self.X_ - x)
        if np.isinf(self.p):
            return d.max(axis=1)
        return (d ** self.p).sum(axis=1) ** (1.0 / self.p)

    def neighbours(self, x):
        d = self._distances(np.asarray(x, float))
        idx = np.argsort(d, kind="stable")[: self.k]
        return idx, d[idx]

    def _predict_one(self, x):
        idx, d = self.neighbours(x)
        w = np.ones(len(idx)) if self.weights == "uniform" else 1.0 / np.maximum(d, 1e-12)
        if self.task == "regression":
            return float(np.average(self.y_[idx], weights=w))
        votes = {}
        for i, wi in zip(idx, w):                      # idx is sorted by distance: nearest first
            votes[self.y_[i]] = votes.get(self.y_[i], 0.0) + wi
        return max(votes, key=votes.get)               # max() keeps the first (nearest) on ties

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.asarray(X, float)])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# The twelve flowers of the lecture (slide 42) and the query (5.2, 3.1)
IRIS12 = np.array([
    [5.3, 3.7], [5.1, 3.8], [7.2, 3.0], [5.4, 3.4], [5.1, 3.3], [5.4, 3.9],
    [7.4, 2.8], [6.1, 2.8], [7.3, 2.9], [6.0, 2.7], [5.8, 2.8], [6.3, 2.3]])
IRIS12_LABELS = np.array(["Setosa", "Setosa", "Virginica", "Setosa", "Setosa", "Setosa",
                          "Virginica", "Versicolor", "Virginica", "Versicolor", "Virginica", "Versicolor"])
QUERY = np.array([5.2, 3.1])

if __name__ == "__main__":
    d = [minkowski(QUERY, x) for x in IRIS12]
    rank = np.argsort(np.argsort(d)) + 1
    for x, lab, di, r in zip(IRIS12, IRIS12_LABELS, d, rank):
        print(f"{x[0]:.1f} {x[1]:.1f} {lab:11s} distance {di:.3f} rank {r}")
    for k in (1, 3, 5, 7, 9):
        m = KNN(k=k).fit(IRIS12, IRIS12_LABELS)
        idx, _ = m.neighbours(QUERY)
        print(f"k={k}: neighbours {sorted(idx + 1)}  ->", m.predict(QUERY[None])[0])
    print("Manhattan / Euclid / Chebyshev from (0,0) to (3,4):",
          [minkowski([0, 0], [3, 4], p) for p in (1, 2, np.inf)])
    # regression: average of the k nearest targets
    xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]); ys = np.array([1.0, 2.0, 2.0, 5.0, 6.0])
    print("regression k=2 at x=3.4:", KNN(k=2, task="regression").fit(xs, ys).predict(np.array([[3.4]])))

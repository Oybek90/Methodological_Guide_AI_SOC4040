# ci: run
"""PCA from scratch with NumPy (eigen-decomposition of the covariance matrix)."""
import numpy as np


class PCAScratch:
    """Minimal PCA: centre, covariance (divisor m-1), eigh, sort, truncate."""

    def __init__(self, n_components):
        self.k = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        S = np.cov(X - self.mean_, rowvar=False)      # d x d, divisor m-1
        vals, vecs = np.linalg.eigh(S)                 # ascending order
        order = np.argsort(vals)[::-1]                 # descending
        self.eigenvalues_ = vals[order]
        self.components_ = vecs[:, order[: self.k]].T  # shape (k, d)
        self.explained_variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


if __name__ == "__main__":
    X = np.array([[4, 11], [8, 4], [13, 5], [7, 14]], dtype=float)
    p = PCAScratch(n_components=1).fit(X)
    print("mean            :", p.mean_)
    print("eigenvalues     :", np.round(p.eigenvalues_, 4))
    print("PC1 direction   :", np.round(p.components_[0], 4))
    print("variance ratio  :", np.round(p.explained_variance_ratio_, 4))
    print("scores (1-D)    :", np.round(p.transform(X).ravel(), 4))
    print("reconstruction  :\n", np.round(p.inverse_transform(p.transform(X)), 3))
    err = ((X - p.inverse_transform(p.transform(X))) ** 2).sum()
    print("squared error   :", round(err, 4), "= (m-1)*lambda2 =", round(3 * p.eigenvalues_[1], 4))

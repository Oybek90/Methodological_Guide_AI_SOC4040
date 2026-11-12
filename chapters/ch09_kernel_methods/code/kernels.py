# ci: run
"""Kernels, Gram matrices, a PSD test and a dual coordinate-ascent kernel SVM (NumPy only)."""
from math import comb

import numpy as np


# ----------------------------------------
# Kernel functions: each returns the Gram block K[i, j] = k(A_i, B_j)
# ----------------------------------------
def linear_kernel(A, B):
    return A @ B.T


def poly_kernel(A, B, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * (A @ B.T) + coef0) ** degree


def sq_dists(A, B):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(sq, 0.0)            # guard against tiny negative round-off


def rbf_kernel(A, B, gamma=1.0):
    return np.exp(-gamma * sq_dists(A, B))


def laplacian_kernel(A, B, gamma=1.0):
    l1 = np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
    return np.exp(-gamma * l1)


def sigmoid_kernel(A, B, gamma=1.0, coef0=0.0):
    return np.tanh(gamma * (A @ B.T) + coef0)


# ----------------------------------------
# Feature-map size and the explicit quadratic map
# ----------------------------------------
def poly_feature_dim(n, d):
    """Number of monomials of degree <= d in n variables: C(n + d, d)."""
    return comb(n + d, d)


def phi_quadratic(x):
    """Explicit map of K(x, z) = (x.z)^2 for x in R^2."""
    return np.array([x[0] ** 2, np.sqrt(2.0) * x[0] * x[1], x[1] ** 2])


def phi_quadratic_bias(x):
    """Explicit 6-feature map of K(x, z) = (x.z + 1)^2 for x in R^2."""
    s = np.sqrt(2.0)
    return np.array([x[0] ** 2, x[1] ** 2, s * x[0] * x[1], s * x[0], s * x[1], 1.0])


# ----------------------------------------
# Mercer check on a finite sample: symmetric + smallest eigenvalue >= -tol
# ----------------------------------------
def is_psd(K, tol=1e-8):
    K = np.asarray(K)
    return bool(np.allclose(K, K.T)) and float(np.linalg.eigvalsh((K + K.T) / 2).min()) >= -tol


# ----------------------------------------
# Kernel SVM, dual coordinate ascent. The bias is absorbed: K' = K + 1, f(x) = sum_i a_i y_i K'(x_i, x)
# ----------------------------------------
class KernelSVM:
    """Soft-margin kernel SVM, labels in {-1, +1}; one multiplier updated at a time."""

    def __init__(self, kernel=rbf_kernel, C=1.0, max_epochs=300, tol=1e-6, seed=0, **kernel_params):
        self.kernel, self.C, self.max_epochs, self.tol, self.seed = kernel, C, max_epochs, tol, seed
        self.kernel_params = kernel_params

    def _K(self, A, B):
        return self.kernel(A, B, **self.kernel_params) + 1.0

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)               # per-fit generator: reproducible
        n = len(y)
        K = self._K(X, X)
        alpha, f, diag = np.zeros(n), np.zeros(n), np.diag(K).copy()
        for epoch in range(self.max_epochs):
            largest = 0.0
            for i in rng.permutation(n):
                new = np.clip(alpha[i] + (1.0 - y[i] * f[i]) / diag[i], 0.0, self.C)
                delta = new - alpha[i]
                if delta != 0.0:
                    f += delta * y[i] * K[:, i]
                    alpha[i] = new
                    largest = max(largest, abs(delta))
            if largest < self.tol:
                break
        self.dual_ = alpha.sum() - 0.5 * (alpha * y) @ K @ (alpha * y)
        sv = alpha > 1e-8
        self.alpha_, self.X_sv_, self.y_sv_, self.n_epochs_ = alpha[sv], X[sv], y[sv], epoch + 1
        return self

    def decision_function(self, X):
        return self._K(X, self.X_sv_) @ (self.alpha_ * self.y_sv_)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


if __name__ == "__main__":
    x, z = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    print("explicit:", phi_quadratic(x) @ phi_quadratic(z), " kernel:", (x @ z) ** 2)
    print("dims C(n+d,d):", [(n, d, poly_feature_dim(n, d)) for n, d in [(2, 2), (100, 2), (100, 5), (784, 3), (784, 5)]])

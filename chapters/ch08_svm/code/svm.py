# ci: run
"""Support vector machines from scratch: primal sub-gradient descent (linear) and a simplified SMO (kernels)."""
import numpy as np


# ----------------------------------------
# Kernels
# ----------------------------------------
def linear_kernel(A, B):
    return A @ B.T


def poly_kernel(A, B, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * (A @ B.T) + coef0) ** degree


def rbf_kernel(A, B, gamma=1.0):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


# ----------------------------------------
# Linear soft-margin SVM: minimise  1/2 ||w||^2 + C * sum_i max(0, 1 - y_i (w.x_i + b))
# ----------------------------------------
class LinearSVM:
    """Stochastic sub-gradient descent on the primal objective (labels in {-1, +1})."""

    def __init__(self, C=1.0, epochs=200, lr=0.01, seed=0):
        self.C, self.epochs, self.lr, self.seed = C, epochs, lr, seed

    def objective(self, X, y):
        hinge = np.maximum(0.0, 1.0 - y * (X @ self.w_ + self.b_))
        return 0.5 * self.w_ @ self.w_ + self.C * hinge.sum()

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.w_, self.b_ = np.zeros(X.shape[1]), 0.0
        for epoch in range(self.epochs):
            eta = self.lr / (1.0 + 0.05 * epoch)                  # decaying step size
            for i in rng.permutation(n):
                if y[i] * (X[i] @ self.w_ + self.b_) >= 1.0:      # outside the margin: only the regulariser acts
                    gw, gb = self.w_ / n, 0.0
                else:                                             # margin violator: the hinge term is active
                    gw, gb = self.w_ / n - self.C * y[i] * X[i], -self.C * y[i]
                self.w_ -= eta * gw
                self.b_ -= eta * gb
        return self

    def decision_function(self, X):
        return X @ self.w_ + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# ----------------------------------------
# Kernel SVM in the dual:  max sum a_i - 1/2 sum a_i a_j y_i y_j K_ij   s.t. 0 <= a_i <= C, sum a_i y_i = 0
# ----------------------------------------
class KernelSVM:
    """Simplified SMO: repeatedly optimise two multipliers analytically (labels in {-1, +1})."""

    def __init__(self, C=1.0, kernel=rbf_kernel, tol=1e-3, max_passes=10, seed=0, **kernel_params):
        self.C, self.kernel, self.tol, self.max_passes, self.seed = C, kernel, tol, max_passes, seed
        self.kernel_params = kernel_params

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        K = self.kernel(X, X, **self.kernel_params)
        a, b, passes = np.zeros(n), 0.0, 0
        f = lambda i: (a * y) @ K[:, i] + b
        while passes < self.max_passes:
            changed = 0
            for i in range(n):
                Ei = f(i) - y[i]
                if (y[i] * Ei < -self.tol and a[i] < self.C) or (y[i] * Ei > self.tol and a[i] > 0):
                    j = int(rng.integers(n - 1))
                    j += j >= i                                    # random j != i
                    Ej = f(j) - y[j]
                    ai_old, aj_old = a[i], a[j]
                    if y[i] != y[j]:
                        L, H = max(0.0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
                    else:
                        L, H = max(0.0, ai_old + aj_old - self.C), min(self.C, ai_old + aj_old)
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if L == H or eta >= 0:
                        continue
                    a[j] = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(a[j] - aj_old) < 1e-5:
                        continue
                    a[i] = ai_old + y[i] * y[j] * (aj_old - a[j])
                    b1 = b - Ei - y[i] * (a[i] - ai_old) * K[i, i] - y[j] * (a[j] - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (a[i] - ai_old) * K[i, j] - y[j] * (a[j] - aj_old) * K[j, j]
                    b = b1 if 0 < a[i] < self.C else b2 if 0 < a[j] < self.C else 0.5 * (b1 + b2)
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
        sv = a > 1e-8
        self.alpha_, self.b_, self.X_sv_, self.y_sv_ = a[sv], b, X[sv], y[sv]
        self.n_support_ = int(sv.sum())
        return self

    def decision_function(self, X):
        K = self.kernel(X, self.X_sv_, **self.kernel_params)
        return K @ (self.alpha_ * self.y_sv_) + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


if __name__ == "__main__":
    # the two-point example of slide 13: x+ = (3,3), x- = (1,1); the exact solution is w = (0.5,0.5), b = -2
    X = np.array([[3.0, 3.0], [1.0, 1.0]])
    y = np.array([1, -1])
    m = KernelSVM(C=1e3, kernel=linear_kernel, max_passes=50).fit(X, y)
    w = (m.alpha_ * m.y_sv_) @ m.X_sv_
    print("alpha =", np.round(m.alpha_, 3), " w =", np.round(w, 3), " b =", round(m.b_, 3),
          " margin width 2/|w| =", round(2 / np.linalg.norm(w), 3))
    print("f(2,1) =", round(float(m.decision_function(np.array([[2.0, 1.0]]))[0]), 3),
          " f(2,2.5) =", round(float(m.decision_function(np.array([[2.0, 2.5]]))[0]), 3))

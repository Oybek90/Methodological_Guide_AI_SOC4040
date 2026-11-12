# ci: run
"""Multiple linear regression: matrix form, gradient descent, normal equation, conditioning."""
import numpy as np

# The 5 x 3 data set of the lecture: columns x1, x2, x3 and the target y.
DATA = np.array([[10, 12, 5, 50],
                 [15, 11, 7, 59],
                 [20, 23, 3, 68],
                 [25, 28, 6, 77],
                 [30, 31, 6, 86]], dtype=float)
X_RAW, Y = DATA[:, :3], DATA[:, 3:]            # y as a column vector (m, 1)


def add_bias(X):
    """Prepend the constant feature x0 = 1."""
    return np.hstack([np.ones((len(X), 1)), X])


def loss(A, y, theta):
    """J(theta) = 1/(2m) * ||A theta - y||^2"""
    e = A @ theta - y
    return float((e.T @ e).item() / (2 * len(y)))


def gradient_descent(A, y, lr, iters):
    """Batch gradient descent, theta <- theta - lr/m * A^T (A theta - y). Returns (theta, losses)."""
    m = len(y)
    theta = np.zeros((A.shape[1], 1))
    hist = np.empty(iters)
    for k in range(iters):
        err = A @ theta - y
        hist[k] = float((err.T @ err).item()) / (2 * m)
        theta = theta - lr * (A.T @ err) / m
    return theta, hist


def normal_equation(A, y):
    """Solve (A^T A) theta = A^T y without forming the inverse explicitly."""
    return np.linalg.solve(A.T @ A, A.T @ y)


def standardise(X, mean=None, std=None):
    """z-score; pass the training mean/std when transforming test data."""
    mean = X.mean(axis=0) if mean is None else mean
    std = X.std(axis=0) if std is None else std
    return (X - mean) / std, mean, std


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    A = add_bias(X_RAW)
    print("A shape", A.shape, " initial loss J(0) =", loss(A, Y, np.zeros((4, 1))))
    th_inv = np.linalg.inv(A.T @ A) @ A.T @ Y
    th_ne = normal_equation(A, Y)
    print("normal equation (inv)  :", th_inv.ravel())
    print("normal equation (solve):", th_ne.ravel())
    print("cond(A^T A) =", f"{np.linalg.cond(A.T @ A):.3e}")
    new = np.array([[1, 10, 12, 5]])
    print("prediction for (10,12,5):", float((new @ th_ne).item()))
    th_raw, h_raw = gradient_descent(A, Y, lr=0.001, iters=100_000)
    print("GD raw, lr=1e-3, 1e5 it:", th_raw.ravel(), " J =", f"{h_raw[-1]:.4f}")
    for lr in (0.002, 0.01):
        _, h = gradient_descent(A, Y, lr=lr, iters=200)
        print(f"GD raw lr={lr}: J after 200 = {h[-1]:.3e}")
    Z, mu, sd = standardise(X_RAW)
    Az = add_bias(Z)
    th_z, h_z = gradient_descent(Az, Y, lr=0.1, iters=500)
    print("GD standardised lr=0.1, 500 it: J =", f"{h_z[-1]:.3e}")
    # back to original units
    w = th_z[1:, 0] / sd
    b = th_z[0, 0] - float(np.sum(th_z[1:, 0] * mu / sd))
    print("back-transformed theta:", np.concatenate([[b], w]))
    print("cond(Az^T Az) =", f"{np.linalg.cond(Az.T @ Az):.3e}")

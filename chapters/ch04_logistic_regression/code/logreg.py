# ci: run
"""Logistic regression from scratch: sigmoid, cross-entropy, gradient descent, L2, one-vs-rest."""
import numpy as np

# The five-example data set of the lecture (slides 18-30): x0 = 1, x1, x2 and the label y.
X_LEC = np.array([[1, 2, 3], [1, 3, 2], [1, 3, 1], [1, 1, 3], [1, 0, 4]], dtype=float)
Y_LEC = np.array([[1], [0], [0], [1], [1]], dtype=float)


def add_bias(X):
    """Prepend the constant feature x0 = 1."""
    return np.hstack([np.ones((len(X), 1)), X])


def sigmoid(z):
    """g(z) = 1/(1+exp(-z)), written so that exp never overflows for large |z|."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def cost(A, y, theta, lam=0.0):
    """Cross-entropy J(theta) plus the L2 term lam/(2m)*sum_{j>=1} theta_j^2."""
    m = len(y)
    z = A @ theta
    # log(1+e^z) - y z is the same quantity as -[y log h + (1-y) log(1-h)] but never gives log(0)
    ce = np.mean(np.logaddexp(0.0, z) - y * z)
    return float(ce + lam / (2 * m) * np.sum(theta[1:] ** 2))


def gradient(A, y, theta, lam=0.0):
    """(1/m) A^T (h - y), with (lam/m) theta_j added for j >= 1 (the bias is not penalised)."""
    m = len(y)
    g = A.T @ (sigmoid(A @ theta) - y) / m
    g[1:] += lam / m * theta[1:]
    return g


def fit(A, y, lr=0.1, iters=1000, lam=0.0):
    """Batch gradient descent from theta = 0. Returns (theta, history of J)."""
    theta = np.zeros((A.shape[1], 1))
    hist = [cost(A, y, theta, lam)]
    for _ in range(iters):
        theta = theta - lr * gradient(A, y, theta, lam)
        hist.append(cost(A, y, theta, lam))
    return theta, np.array(hist)


def predict_proba(A, theta):
    return sigmoid(A @ theta)


def predict(A, theta, threshold=0.5):
    return (predict_proba(A, theta) >= threshold).astype(int)


def fit_one_vs_rest(A, labels, n_classes, lr=0.1, iters=1000, lam=0.0):
    """One binary classifier per class; row c of the returned matrix is theta_c."""
    thetas = []
    for c in range(n_classes):
        yc = (labels == c).astype(float).reshape(-1, 1)
        t, _ = fit(A, yc, lr, iters, lam)
        thetas.append(t.ravel())
    return np.array(thetas)


def predict_one_vs_rest(A, thetas):
    """Pick the class whose classifier is most confident: argmax_c h_c(x)."""
    return np.argmax(sigmoid(A @ thetas.T), axis=1)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_softmax(A, labels, n_classes, lr=0.1, iters=1000, lam=0.0):
    """Multinomial logistic regression: Theta has shape (n+1, K); gradient A^T (P - Y)/m."""
    m = len(labels)
    Y = np.eye(n_classes)[labels]
    Theta = np.zeros((A.shape[1], n_classes))
    for _ in range(iters):
        G = A.T @ (softmax(A @ Theta) - Y) / m
        G[1:] += lam / m * Theta[1:]
        Theta -= lr * G
    return Theta


if __name__ == "__main__":
    # Reproduce the hand calculation of the lecture: theta = 0 gives h = 0.5 and J = 0.693.
    theta = np.zeros((3, 1))
    print("h at theta=0:", predict_proba(X_LEC, theta).ravel(), " J =", round(cost(X_LEC, Y_LEC, theta), 3))
    print("gradient at 0:", gradient(X_LEC, Y_LEC, theta).ravel())
    for k in range(1, 4):
        theta = theta - 0.1 * gradient(X_LEC, Y_LEC, theta)
        print(f"iteration {k}: theta =", np.round(theta.ravel(), 3), " J =", round(cost(X_LEC, Y_LEC, theta), 3))
    theta, hist = fit(X_LEC, Y_LEC, lr=0.1, iters=5000)
    print("after 5000 its: theta =", np.round(theta.ravel(), 2), " J =", round(hist[-1], 4),
          " predictions:", predict(X_LEC, theta).ravel())

# ci: run
"""Linear regression with several variables on synthetic house data:
gradient descent with and without feature scaling, the normal equation, and scikit-learn."""
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
m = 200
size = rng.uniform(50, 250, m)            # m^2
rooms = rng.integers(1, 6, m).astype(float)
age = rng.uniform(0, 50, m)               # years
price = 20 + 1.5 * size + 8.0 * rooms - 0.6 * age + rng.normal(0, 10, m)   # thousands of $
X = np.column_stack([size, rooms, age])


def add_bias(A):
    return np.column_stack([np.ones(len(A)), A])


def gd(A, y, alpha, steps):
    """Vectorised batch gradient descent: theta <- theta - alpha/m * A^T (A theta - y)."""
    theta = np.zeros(A.shape[1])
    hist = []
    for _ in range(steps):
        err = A @ theta - y
        hist.append(float(err @ err / (2 * len(y))))
        theta = theta - alpha * (A.T @ err) / len(y)
    return theta, np.array(hist)


# Normal equation: solve (A^T A) theta = A^T y  (exact least squares)
A = add_bias(X)
theta_ne = np.linalg.solve(A.T @ A, A.T @ price)

# Gradient descent on standardised features
mu, sd = X.mean(axis=0), X.std(axis=0)
As = add_bias((X - mu) / sd)
theta_s, hist_s = gd(As, price, alpha=0.1, steps=200)
# back-transform to original units
w = theta_s[1:] / sd
b = theta_s[0] - np.sum(theta_s[1:] * mu / sd)
theta_gd = np.concatenate([[b], w])

# Gradient descent on raw features: need a very small alpha to avoid divergence
_, hist_raw = gd(A, price, alpha=1e-5, steps=200)

sk = LinearRegression().fit(X, price)
theta_sk = np.concatenate([[sk.intercept_], sk.coef_])

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("true      [20, 1.5, 8.0, -0.6]")
    print("normal eq.", theta_ne)
    print("GD scaled ", theta_gd)
    print("sklearn   ", theta_sk)
    print("J after 200 steps: scaled", round(hist_s[-1], 2), " raw (alpha=1e-5)", round(hist_raw[-1], 2))
    print("steps for scaled GD to reach J < 1.05 * J_min:",
          int(np.argmax(hist_s < 1.05 * hist_s[-1])))
    print("max |difference| NE vs GD vs sklearn:", float(np.abs(theta_ne - theta_gd).max()),
          float(np.abs(theta_ne - theta_sk).max()))

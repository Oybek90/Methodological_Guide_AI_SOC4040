# ci: run
"""Linear regression with one variable: hypothesis, cost, gradient, gradient descent."""
import numpy as np


def hypothesis(theta0, theta1, x):
    """h_theta(x) = theta0 + theta1 * x"""
    return theta0 + theta1 * np.asarray(x, dtype=float)


def cost(theta0, theta1, x, y):
    """Squared-error cost J = 1/(2m) * sum (h(x_i) - y_i)^2"""
    err = hypothesis(theta0, theta1, x) - np.asarray(y, dtype=float)
    return float(np.sum(err ** 2) / (2 * len(err)))


def gradient(theta0, theta1, x, y):
    """Partial derivatives (dJ/dtheta0, dJ/dtheta1)."""
    x = np.asarray(x, dtype=float)
    err = hypothesis(theta0, theta1, x) - np.asarray(y, dtype=float)
    return float(err.mean()), float((err * x).mean())


def gradient_descent(x, y, alpha, steps, theta0=0.0, theta1=0.0):
    """Batch gradient descent. Returns the path [(theta0, theta1, J), ...]."""
    path = [(theta0, theta1, cost(theta0, theta1, x, y))]
    for _ in range(steps):
        g0, g1 = gradient(theta0, theta1, x, y)
        # simultaneous update: both gradients were computed before either theta changed
        theta0, theta1 = theta0 - alpha * g0, theta1 - alpha * g1
        path.append((theta0, theta1, cost(theta0, theta1, x, y)))
    return path


def gradient_descent_1d(x, y, alpha, steps, theta1=0.0):
    """Simplified model h(x) = theta1 * x (theta0 fixed at 0), as in the lecture's toy example."""
    x = np.asarray(x, dtype=float)
    path = [(theta1, cost(0.0, theta1, x, y))]
    for _ in range(steps):
        theta1 = theta1 - alpha * gradient(0.0, theta1, x, y)[1]
        path.append((theta1, cost(0.0, theta1, x, y)))
    return path


if __name__ == "__main__":
    # Toy set of the lecture: (1,1), (2,2), (3,3).
    x, y = [1, 2, 3], [1, 2, 3]
    for t1 in (1.0, 0.5, 0.0, 1.5, 2.0):
        print(f"J(theta0=0, theta1={t1}) = {cost(0.0, t1, x, y):.4f}")
    p = gradient_descent_1d(x, y, alpha=0.1, steps=100, theta1=2.5)
    print("toy GD (theta0 = 0) after 100 steps: theta1 =", round(p[-1][0], 4), " J =", f"{p[-1][1]:.2e}")
    p = gradient_descent(x, y, alpha=0.1, steps=100)
    print("toy GD (both parameters) after 100 steps:", [round(v, 4) for v in p[-1]])

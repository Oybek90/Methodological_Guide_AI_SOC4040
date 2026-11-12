# ci: run
"""Polynomial regression from scratch: features, least squares, ridge, MSE and R^2."""
import numpy as np


def true_curve(x):
    return 0.5 * x ** 3 - x ** 2 + 2 * x + 5


def make_data(n, noise, rng):
    """Synthetic data y = 0.5 x^3 - x^2 + 2x + 5 + noise on x in [-3, 3]."""
    x = rng.uniform(-3, 3, n)
    return x, true_curve(x) + rng.normal(0, noise, n)


def poly_features(x, degree):
    """Columns [1, x, x^2, ..., x^degree]."""
    return np.column_stack([x ** d for d in range(degree + 1)])


def fit_ols(A, y):
    """Least squares by SVD (numerically safer than inverting A^T A)."""
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fit_ridge(A, y, lam):
    """Ridge: minimise ||A w - y||^2 + lam * ||w[1:]||^2 (the bias w[0] is not penalised)."""
    P = np.eye(A.shape[1])
    P[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + lam * P, A.T @ y)


def mse(y, p):
    return float(np.mean((y - p) ** 2))


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def split_fit(degree, n=30, noise=3.0, seed=0, lam=None):
    """Train on n points, test on 500 fresh points; features are standardised with train statistics."""
    rng = np.random.default_rng(seed)
    xtr, ytr = make_data(n, noise, rng)
    xte, yte = make_data(500, noise, rng)
    Atr, Ate = poly_features(xtr, degree), poly_features(xte, degree)
    if degree > 0:
        mu, sd = Atr[:, 1:].mean(axis=0), Atr[:, 1:].std(axis=0)
        Atr[:, 1:], Ate[:, 1:] = (Atr[:, 1:] - mu) / sd, (Ate[:, 1:] - mu) / sd
    w = fit_ols(Atr, ytr) if lam is None else fit_ridge(Atr, ytr, lam)
    return mse(ytr, Atr @ w), mse(yte, Ate @ w)


if __name__ == "__main__":
    # 1. degree sweep, mean over 200 random training sets
    print("degree | train MSE | test MSE   (n=30, noise sd=3, noise variance=9)")
    for d in (1, 2, 3, 5, 9, 12):
        r = np.array([split_fit(d, seed=s) for s in range(200)])
        print(f"{d:6d} | {r[:, 0].mean():9.2f} | {np.median(r[:, 1]):8.2f} (median)  {r[:, 1].mean():10.2f} (mean)")
    # 2. ridge on degree 9
    print("ridge, degree 9:")
    for lam in (0, 0.01, 0.1, 1, 10, 100):
        r = np.array([split_fit(9, seed=s, lam=lam) for s in range(200)])
        print(f"  lambda={lam:<6} train {r[:, 0].mean():6.2f}  test median {np.median(r[:, 1]):6.2f}  mean {r[:, 1].mean():8.2f}")
    # 3. cross-check with scikit-learn
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    rng = np.random.default_rng(1)
    x, y = make_data(40, 3.0, rng)
    mine = fit_ols(poly_features(x, 3), y)
    sk = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(x[:, None], y)
    xs = np.linspace(-3, 3, 7)
    print("max |scratch - sklearn| on a grid:", float(np.abs(poly_features(xs, 3) @ mine - sk.predict(xs[:, None])).max()))
    print("cond(A^T A) for degree 3, 9 on raw x:", [f"{np.linalg.cond(poly_features(x, d).T @ poly_features(x, d)):.2e}" for d in (3, 9)])

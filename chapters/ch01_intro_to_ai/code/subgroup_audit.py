# ci: run
"""Why 'overall accuracy' hides unfairness: a synthetic two-group audit."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def make_group(n, shift, rng):
    """Two features; the label boundary depends on the group (different 'shift')."""
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] + shift * X[:, 1] > 0).astype(int)
    return X, y


def features(X, g, aware):
    """With aware=True the model also sees the group and its interaction with the inputs."""
    if not aware:
        return X
    return np.c_[X, np.full(len(X), g), g * X]


def experiment(seed, n_a=2000, n_b=100, n_test=2000, balanced=False, aware=False):
    rng = np.random.default_rng(seed)
    n_b_train = n_a if balanced else n_b
    Xa, ya = make_group(n_a, shift=0.0, rng=rng)           # majority group A (g = 0)
    Xb, yb = make_group(n_b_train, shift=3.0, rng=rng)     # minority group B (g = 1), other pattern
    Xtr = np.vstack([features(Xa, 0, aware), features(Xb, 1, aware)])
    clf = LogisticRegression(max_iter=1000).fit(Xtr, np.r_[ya, yb])
    Xta, yta = make_group(n_test, 0.0, rng)
    Xtb, ytb = make_group(n_test, 3.0, rng)
    acc_a = clf.score(features(Xta, 0, aware), yta)
    acc_b = clf.score(features(Xtb, 1, aware), ytb)
    overall = (acc_a * n_a + acc_b * n_b_train) / (n_a + n_b_train)   # accuracy weighted by the training mix
    return acc_a, acc_b, overall


SETTINGS = {
    "imbalanced data, one model": dict(balanced=False, aware=False),
    "balanced data, one model": dict(balanced=True, aware=False),
    "imbalanced data, group-aware model": dict(balanced=False, aware=True),
}

if __name__ == "__main__":
    for name, kw in SETTINGS.items():
        m = np.array([experiment(s, **kw) for s in range(20)]).mean(axis=0)
        print(f"{name:36s} (20 seeds): group A {m[0]:.3f} | group B {m[1]:.3f} | overall {m[2]:.3f}")

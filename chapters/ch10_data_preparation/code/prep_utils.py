# ci: run
"""Data-preparation building blocks written from scratch (NumPy / pandas), checked against scikit-learn."""
import numpy as np
import pandas as pd


# ----------------------------------------
# Outlier rules
# ----------------------------------------
def iqr_bounds(x, k=1.5):
    q1, q3 = np.nanpercentile(x, [25, 75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def iqr_outliers(x, k=1.5):
    """Boolean mask of values outside [Q1 - k*IQR, Q3 + k*IQR] (NaN is never an outlier)."""
    lo, hi = iqr_bounds(x, k)
    x = np.asarray(x, dtype=float)
    return (x < lo) | (x > hi)


def zscore_outliers(x, thr=3.0):
    x = np.asarray(x, dtype=float)
    return np.abs((x - np.nanmean(x)) / np.nanstd(x)) > thr


def iqr_filter_rows(df, k=1.5):
    """Keep rows that are inside the IQR fences in EVERY column (the .all(axis=1) the slide forgets)."""
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    inside = (df >= q1 - k * iqr) & (df <= q3 + k * iqr)
    return df[inside.all(axis=1)]


# ----------------------------------------
# Scalers with the fit / transform contract
# ----------------------------------------
class MinMax:
    def fit(self, X):
        X = np.asarray(X, float); self.lo_, self.hi_ = X.min(0), X.max(0); return self

    def transform(self, X):
        return (np.asarray(X, float) - self.lo_) / (self.hi_ - self.lo_)


class Standard:
    def fit(self, X):
        X = np.asarray(X, float); self.mu_, self.sd_ = X.mean(0), X.std(0); return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mu_) / self.sd_


class Robust:
    def fit(self, X):
        X = np.asarray(X, float)
        self.med_ = np.median(X, 0); q1, q3 = np.percentile(X, [25, 75], axis=0); self.iqr_ = q3 - q1; return self

    def transform(self, X):
        return (np.asarray(X, float) - self.med_) / self.iqr_


# ----------------------------------------
# SMOTE: interpolate between a minority point and one of its k nearest minority neighbours
# ----------------------------------------
def smote(X, y, minority=1, k=5, seed=0):
    rng = np.random.default_rng(seed)
    Xm = X[y == minority]
    n_new = int((y != minority).sum() - len(Xm))
    d = ((Xm[:, None, :] - Xm[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nbr = np.argsort(d, axis=1)[:, :k]
    i = rng.integers(len(Xm), size=n_new)
    j = nbr[i, rng.integers(k, size=n_new)]
    lam = rng.random((n_new, 1))
    Xs = Xm[i] + lam * (Xm[j] - Xm[i])
    return np.vstack([X, Xs]), np.r_[y, np.full(n_new, minority)]


# ----------------------------------------
# Target encoding without leakage: smoothed means, computed out-of-fold on the training set
# ----------------------------------------
def target_encode(cat_train, y_train, cat_test, m=10.0, folds=5, seed=0):
    cat_train, cat_test = pd.Series(cat_train).reset_index(drop=True), pd.Series(cat_test).reset_index(drop=True)
    y = pd.Series(np.asarray(y_train, float))
    prior = y.mean()

    def fit_map(c, t):
        g = t.groupby(c).agg(["sum", "count"])
        return (g["sum"] + m * prior) / (g["count"] + m)

    enc = np.full(len(cat_train), prior)
    idx = np.random.default_rng(seed).permutation(len(cat_train))
    for part in np.array_split(idx, folds):
        rest = np.setdiff1d(idx, part)
        mp = fit_map(cat_train.iloc[rest], y.iloc[rest])
        enc[part] = cat_train.iloc[part].map(mp).fillna(prior).to_numpy()
    full = fit_map(cat_train, y)
    return enc, cat_test.map(full).fillna(prior).to_numpy()


# ----------------------------------------
# Chronological split for time series
# ----------------------------------------
def time_split(df, time_col, test_frac=0.2):
    df = df.sort_values(time_col)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    X = np.random.default_rng(0).lognormal(size=(100, 3))
    for mine, ref in ((MinMax(), MinMaxScaler()), (Standard(), StandardScaler()), (Robust(), RobustScaler())):
        assert np.allclose(mine.fit(X).transform(X), ref.fit_transform(X)), type(mine).__name__
    print("scalers agree with scikit-learn")

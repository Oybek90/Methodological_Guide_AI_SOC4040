# ci: run
"""Evaluation metrics from scratch: regression, classification, ROC/PR curves, k-fold splitting."""
import numpy as np

# ------------------------------------------------------------------ regression
def mse(y, h):
    return float(np.mean((np.asarray(y, float) - np.asarray(h, float)) ** 2))


def rmse(y, h):
    return float(np.sqrt(mse(y, h)))


def mae(y, h):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(h, float))))


def r2(y, h):
    """1 - SS_res / SS_tot; 0 for the constant mean predictor, negative if worse than that."""
    y, h = np.asarray(y, float), np.asarray(h, float)
    return float(1.0 - np.sum((y - h) ** 2) / np.sum((y - y.mean()) ** 2))


# ------------------------------------------------------------------ classification
def confusion(y, pred):
    """Counts (TP, FP, FN, TN) for labels in {0, 1}; the positive class is 1."""
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return tp, fp, fn, tn


def _div(a, b):
    return a / b if b > 0 else 0.0        # the convention of the lecture notebook: 0 when undefined


def accuracy(y, pred):
    tp, fp, fn, tn = confusion(y, pred)
    return _div(tp + tn, tp + fp + fn + tn)


def precision(y, pred):
    tp, fp, _, _ = confusion(y, pred)
    return _div(tp, tp + fp)


def recall(y, pred):
    tp, _, fn, _ = confusion(y, pred)
    return _div(tp, tp + fn)


def specificity(y, pred):
    _, fp, _, tn = confusion(y, pred)
    return _div(tn, tn + fp)


def fbeta(y, pred, beta=1.0):
    """F_beta = (1+b^2) P R / (b^2 P + R); beta > 1 weights recall more, beta < 1 precision."""
    p, r = precision(y, pred), recall(y, pred)
    return _div((1 + beta ** 2) * p * r, beta ** 2 * p + r)


def macro_average(metric, y, pred, classes):
    """Unweighted mean of a one-vs-rest metric over the classes."""
    return float(np.mean([metric((np.asarray(y) == c).astype(int), (np.asarray(pred) == c).astype(int))
                          for c in classes]))


def micro_precision_recall(y, pred, classes):
    """Pool TP, FP, FN over all classes first (for single-label problems this equals accuracy)."""
    tp = fp = fn = 0
    for c in classes:
        a, b, c_, _ = confusion((np.asarray(y) == c).astype(int), (np.asarray(pred) == c).astype(int))
        tp, fp, fn = tp + a, fp + b, fn + c_
    return _div(tp, tp + fp), _div(tp, tp + fn)


# ------------------------------------------------------------------ threshold-free curves
def roc_curve(y, score):
    """False-positive rate and true-positive rate for every distinct threshold (descending)."""
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(-score, kind="stable")
    y, score = y[order], score[order]
    distinct = np.r_[np.where(np.diff(score))[0], len(y) - 1]      # last index of every score value
    tps = np.cumsum(y)[distinct]
    fps = 1 + distinct - tps
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]]


def auc(x, y):
    """Trapezoidal area under a curve."""
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def precision_recall_curve(y, score):
    y, score = np.asarray(y), np.asarray(score)
    order = np.argsort(-score, kind="stable")
    y, score = y[order], score[order]
    distinct = np.r_[np.where(np.diff(score))[0], len(y) - 1]
    tps = np.cumsum(y)[distinct]
    prec = tps / (1 + distinct)
    rec = tps / tps[-1]
    return prec, rec


def average_precision(y, score):
    """Step-wise area under the precision-recall curve (the definition used by scikit-learn)."""
    prec, rec = precision_recall_curve(y, score)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


# ------------------------------------------------------------------ resampling
def kfold_indices(n, k, seed=0):
    """Shuffle 0..n-1 and cut into k folds; yields (train_idx, test_idx)."""
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        yield np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]


def cross_val_score(fit_predict, X, y, k=5, seed=0, metric=accuracy):
    """fit_predict(Xtr, ytr, Xte) -> predictions. Returns the k fold scores."""
    return np.array([metric(y[te], fit_predict(X[tr], y[tr], X[te]))
                     for tr, te in kfold_indices(len(y), k, seed)])


if __name__ == "__main__":
    # Regression example of the lecture notebook
    h = np.array([11, 19, 28, 35, 48, 60, 66, 78, 91, 99])
    y = np.array([10, 20, 30, 40, 40, 60, 70, 80, 90, 100])
    print("MSE %.2f  RMSE %.3f  MAE %.2f  R2 %.4f" % (mse(y, h), rmse(y, h), mae(y, h), r2(y, h)))

    # Classification example of the lecture notebook
    hb = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    yb = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    tp, fp, fn, tn = confusion(yb, hb)
    print("TP FP FN TN =", tp, fp, fn, tn, " accuracy %.2f precision %.3f recall %.2f F1 %.3f"
          % (accuracy(yb, hb), precision(yb, hb), recall(yb, hb), fbeta(yb, hb)))

    # Banana / apple example of the slides (banana = positive): rows = actual, columns = predicted
    y_s = np.array([1] * 6 + [0] * 4)
    p_s = np.array([1] * 4 + [0] * 2 + [1] * 1 + [0] * 3)
    print("slide example: TP FP FN TN =", confusion(y_s, p_s), " precision %.3f recall %.3f F1 %.3f"
          % (precision(y_s, p_s), recall(y_s, p_s), fbeta(y_s, p_s)))

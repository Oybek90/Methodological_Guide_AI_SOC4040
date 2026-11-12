# ci: run
"""Naive Bayes from scratch: categorical, multinomial (word counts) and Gaussian variants."""
import math
from collections import Counter, defaultdict

import numpy as np

# The 15 customers of the lecture (slide 14): (day, discount, free delivery) -> purchase
SHOPPING = [
    ("Weekday", "Yes", "Yes", "Yes"), ("Weekday", "Yes", "Yes", "Yes"), ("Weekday", "No", "No", "No"),
    ("Holiday", "Yes", "Yes", "Yes"), ("Weekend", "Yes", "Yes", "Yes"), ("Holiday", "No", "Yes", "No"),
    ("Weekend", "Yes", "No", "No"), ("Weekday", "Yes", "Yes", "Yes"), ("Weekend", "Yes", "Yes", "No"),
    ("Holiday", "Yes", "Yes", "Yes"), ("Holiday", "No", "Yes", "Yes"), ("Holiday", "No", "No", "No"),
    ("Weekend", "Yes", "Yes", "Yes"), ("Holiday", "Yes", "Yes", "Yes"), ("Weekday", "No", "No", "Yes"),
]


class CategoricalNB:
    """P(y | x) proportional to P(y) * prod_j P(x_j | y), probabilities estimated by counting.

    alpha = 0 is the plain relative frequency of the lecture; alpha = 1 is Laplace smoothing."""

    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        self.n_ = Counter(y)
        self.m_ = len(y)
        self.values_ = [sorted({row[j] for row in X}) for j in range(len(X[0]))]
        self.count_ = [defaultdict(int) for _ in range(len(X[0]))]  # count_[j][(value, class)]
        for row, c in zip(X, y):
            for j, v in enumerate(row):
                self.count_[j][(v, c)] += 1
        return self

    def prior(self, c):
        return self.n_[c] / self.m_

    def likelihood(self, j, v, c):
        k = len(self.values_[j])
        return (self.count_[j][(v, c)] + self.alpha) / (self.n_[c] + self.alpha * k)

    def joint(self, x, c):
        """Unnormalised score P(y = c) * prod_j P(x_j | y = c)."""
        p = self.prior(c)
        for j, v in enumerate(x):
            p *= self.likelihood(j, v, c)
        return p

    def predict_proba(self, x):
        scores = {c: self.joint(x, c) for c in self.classes_}
        total = sum(scores.values())
        return {c: s / total for c, s in scores.items()}

    def predict(self, x):
        p = self.predict_proba(x)
        return max(p, key=p.get)


class MultinomialNB:
    """Naive Bayes for word counts, with Laplace smoothing alpha, computed with logarithms."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, docs, y):
        self.classes_ = sorted(set(y))
        self.vocab_ = sorted({w for d in docs for w in d.split()})
        self.logprior_ = {c: math.log(y.count(c) / len(y)) for c in self.classes_}
        self.counts_ = {c: Counter() for c in self.classes_}
        for d, c in zip(docs, y):
            self.counts_[c].update(d.split())
        self.total_ = {c: sum(self.counts_[c].values()) for c in self.classes_}
        return self

    def log_scores(self, doc):
        V = len(self.vocab_)
        out = {}
        for c in self.classes_:
            s = self.logprior_[c]
            for w in doc.split():
                if w not in self.vocab_:
                    continue                                   # unknown words are ignored
                p = (self.counts_[c][w] + self.alpha) / (self.total_[c] + self.alpha * V)
                s += math.log(p) if p > 0 else -math.inf
            out[c] = s
        return out

    def predict_proba(self, doc):
        ls = self.log_scores(doc)
        m = max(ls.values())
        e = {c: math.exp(v - m) if v > -math.inf else 0.0 for c, v in ls.items()}
        z = sum(e.values())
        if z == 0:                                             # every class has probability 0
            return {c: float("nan") for c in e}
        return {c: v / z for c, v in e.items()}


class GaussianNB:
    """Each feature is Gaussian inside each class; var_smoothing is added to every variance."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.prior_ = np.array([np.mean(y == c) for c in self.classes_])
        self.mean_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        eps = self.var_smoothing * X.var(axis=0).max()
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_]) + eps
        return self

    def log_joint(self, X):
        X = np.asarray(X, float)
        out = []
        for k in range(len(self.classes_)):
            ll = -0.5 * (np.log(2 * np.pi * self.var_[k]) + (X - self.mean_[k]) ** 2 / self.var_[k]).sum(axis=1)
            out.append(np.log(self.prior_[k]) + ll)
        return np.array(out).T

    def predict_proba(self, X):
        lj = self.log_joint(X)
        lj -= lj.max(axis=1, keepdims=True)
        p = np.exp(lj)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.log_joint(X), axis=1)]


if __name__ == "__main__":
    X = [r[:3] for r in SHOPPING]
    y = [r[3] for r in SHOPPING]
    nb = CategoricalNB(alpha=0).fit(X, y)
    q = ("Holiday", "Yes", "Yes")
    print("P(No), P(Yes) =", nb.prior("No"), nb.prior("Yes"))
    s_no, s_yes = nb.joint(q, "No"), nb.joint(q, "Yes")
    print(f"numerators: No {s_no:.4f}  Yes {s_yes:.4f}")
    # the lecture divides by P(Holiday) P(Discount=Yes) P(FD=Yes), which assumes independent features
    evid = (6 / 15) * (10 / 15) * (11 / 15)
    print(f"slide values: No {s_no / evid:.3f}  Yes {s_yes / evid:.3f}  sum {(s_no + s_yes) / evid:.3f}")
    p = nb.predict_proba(q)
    print(f"normalised: No {p['No']:.3f}  Yes {p['Yes']:.3f}  ->", nb.predict(q))
    # the same model for a Weekend customer without discount and without free delivery
    q2 = ("Weekend", "No", "No")
    print("weekend/no/no:", {k: round(v, 3) for k, v in nb.predict_proba(q2).items()})

    docs = ["win money now", "win a prize now", "cheap money offer", "meeting at noon", "lunch at noon", "project meeting now"]
    lab = ["spam", "spam", "spam", "ham", "ham", "ham"]
    for a in (0.0, 1.0):
        m = MultinomialNB(alpha=a).fit(docs, lab)
        print(f"alpha={a}:", {k: round(v, 3) for k, v in m.predict_proba("win meeting").items()})

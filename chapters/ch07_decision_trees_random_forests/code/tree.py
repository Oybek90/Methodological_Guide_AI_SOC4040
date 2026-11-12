# ci: run
"""Decision trees from scratch: entropy, Gini, information gain, ID3 and a binary CART-style tree."""
import math
from collections import Counter

import numpy as np

# ----------------------------------------
# Impurity measures
# ----------------------------------------
def entropy(labels):
    """E(S) = - sum_c p_c log2 p_c   (0 log 0 = 0)."""
    n = len(labels)
    return -sum((k / n) * math.log2(k / n) for k in Counter(labels).values())


def gini(labels):
    """G(S) = 1 - sum_c p_c^2."""
    n = len(labels)
    return 1.0 - sum((k / n) ** 2 for k in Counter(labels).values())


def information_gain(rows, feature, target, impurity=entropy):
    """IG(S, a) = E(S) - sum_v |S_v| / |S| E(S_v) for a categorical feature a."""
    labels = [r[target] for r in rows]
    gain = impurity(labels)
    for v in {r[feature] for r in rows}:
        sub = [r[target] for r in rows if r[feature] == v]
        gain -= len(sub) / len(rows) * impurity(sub)
    return gain


# ----------------------------------------
# ID3: multiway splits on categorical features
# ----------------------------------------
PLAY_TENNIS = [  # slide 32 and the notebook: Outlook, Temp, Humidity, Wind -> PlayTennis
    ("Sunny", "Hot", "High", "Weak", "No"), ("Sunny", "Hot", "High", "Strong", "No"),
    ("Overcast", "Hot", "High", "Weak", "Yes"), ("Rain", "Mild", "High", "Weak", "Yes"),
    ("Rain", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Cool", "Normal", "Strong", "No"),
    ("Overcast", "Cool", "Normal", "Strong", "Yes"), ("Sunny", "Mild", "High", "Weak", "No"),
    ("Sunny", "Cool", "Normal", "Weak", "Yes"), ("Rain", "Mild", "Normal", "Weak", "Yes"),
    ("Sunny", "Mild", "Normal", "Strong", "Yes"), ("Overcast", "Mild", "High", "Strong", "Yes"),
    ("Overcast", "Hot", "Normal", "Weak", "Yes"), ("Rain", "Mild", "High", "Strong", "No")]
NAMES = ["Outlook", "Temp", "Humidity", "Wind", "PlayTennis"]
PLAY_ROWS = [dict(zip(NAMES, r)) for r in PLAY_TENNIS]


def id3(rows, features, target="PlayTennis"):
    """Return a nested tree: a class label (leaf) or (feature, {value: subtree}, majority_label)."""
    labels = [r[target] for r in rows]
    majority = Counter(labels).most_common(1)[0][0]
    if len(set(labels)) == 1 or not features:
        return majority
    best = max(features, key=lambda f: information_gain(rows, f, target))
    rest = [f for f in features if f != best]
    return (best, {v: id3([r for r in rows if r[best] == v], rest, target) for v in sorted({r[best] for r in rows})}, majority)


def id3_predict(tree, x):
    while isinstance(tree, tuple):
        feature, branches, majority = tree
        tree = branches.get(x[feature], majority)
    return tree


def show(tree, indent=""):
    if not isinstance(tree, tuple):
        return indent + "-> " + tree + "\n"
    out = ""
    for v, sub in tree[1].items():
        out += f"{indent}{tree[0]} = {v}\n" + show(sub, indent + "    ")
    return out


# ----------------------------------------
# CART-style binary tree on numerical features (classification and regression)
# ----------------------------------------
class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n", "impurity", "gain")


class DecisionTree:
    """Binary tree: at each node choose the feature and threshold with the largest impurity decrease.

    criterion: 'gini' or 'entropy' (classification, labels 0..K-1) or 'mse' (regression)."""

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, rng=None):
        self.criterion, self.max_depth = criterion, max_depth
        self.min_samples_split, self.min_samples_leaf = min_samples_split, min_samples_leaf
        self.max_features, self.rng = max_features, rng or np.random.default_rng(0)

    # impurity of a set of targets given as sums: for classification the class counts, for mse (n, sum, sum of squares)
    def _impurity_counts(self, counts):
        n = counts.sum(axis=-1, keepdims=True)
        p = counts / np.maximum(n, 1)
        if self.criterion == "gini":
            return 1.0 - (p ** 2).sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=-1)

    def _node_impurity(self, y):
        if self.criterion == "mse":
            return float(np.var(y))
        return float(self._impurity_counts(np.bincount(y, minlength=self.k_).astype(float)))

    def _best_split(self, X, y, features):
        n, best = len(y), (0.0, None, None)               # (gain, feature, threshold)
        parent = self._node_impurity(y)
        for j in features:
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            valid = np.nonzero(xs[1:] > xs[:-1])[0]                    # boundaries between distinct values
            valid = valid[(valid + 1 >= self.min_samples_leaf) & (n - valid - 1 >= self.min_samples_leaf)]
            if len(valid) == 0:
                continue
            nl, nr = valid + 1.0, n - valid - 1.0
            if self.criterion == "mse":
                ys = y[order].astype(float)
                c1, c2 = np.cumsum(ys), np.cumsum(ys ** 2)
                s1, s2 = c1[valid], c2[valid]
                il = s2 / nl - (s1 / nl) ** 2
                ir = (c2[-1] - s2) / nr - ((c1[-1] - s1) / nr) ** 2
            else:
                onehot = np.eye(self.k_)[y[order]]
                cum = np.cumsum(onehot, axis=0)
                left, right = cum[valid], cum[-1] - cum[valid]
                il, ir = self._impurity_counts(left), self._impurity_counts(right)
            gain = parent - (nl / n) * il - (nr / n) * ir
            i = int(np.argmax(gain))
            if gain[i] > best[0] + 1e-12:
                best = (float(gain[i]), j, 0.5 * (xs[valid[i]] + xs[valid[i] + 1]))
        return best

    def _grow(self, X, y, depth):
        node = Node()
        node.n, node.impurity = len(y), self._node_impurity(y)
        node.feature = node.threshold = node.left = node.right = None
        node.gain = 0.0
        node.value = float(y.mean()) if self.criterion == "mse" else np.bincount(y, minlength=self.k_) / len(y)
        stop = (len(y) < self.min_samples_split or node.impurity <= 1e-12
                or (self.max_depth is not None and depth >= self.max_depth))
        if stop:
            return node
        d = X.shape[1]
        m = d if self.max_features is None else (max(1, int(np.sqrt(d))) if self.max_features == "sqrt" else self.max_features)
        feats = np.arange(d) if m >= d else self.rng.choice(d, m, replace=False)
        gain, j, t = self._best_split(X, y, feats)
        if j is None:
            return node
        mask = X[:, j] <= t
        node.feature, node.threshold, node.gain = j, t, gain
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.k_ = 0 if self.criterion == "mse" else int(y.max()) + 1
        self.n_features_ = X.shape[1]
        self.root_ = self._grow(X, y if self.criterion == "mse" else y.astype(int), 0)
        return self

    def _leaf(self, x):
        node = self.root_
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        return np.array([self._leaf(x).value for x in np.asarray(X, float)])

    def predict(self, X):
        v = self.predict_proba(X)
        return v if self.criterion == "mse" else v.argmax(axis=1)

    def depth(self, node=None):
        node = node or self.root_
        return 0 if node.feature is None else 1 + max(self.depth(node.left), self.depth(node.right))

    def n_leaves(self, node=None):
        node = node or self.root_
        return 1 if node.feature is None else self.n_leaves(node.left) + self.n_leaves(node.right)

    def feature_importances(self):
        """Mean decrease of impurity: each split adds (n_node / n) * gain to its feature; normalised to sum 1."""
        imp = np.zeros(self.n_features_)
        stack = [self.root_]
        while stack:
            nd = stack.pop()
            if nd.feature is not None:
                imp[nd.feature] += nd.n * nd.gain
                stack += [nd.left, nd.right]
        total = imp.sum()
        return imp / total if total > 0 else imp


if __name__ == "__main__":
    print("entropy(5 yes, 2 no)  = %.3f   (slide 26 rounds to 0.85)" % entropy(["y"] * 5 + ["n"] * 2))
    print("entropy(3 yes, 2 no)  = %.3f   (slide 27 rounds to 0.966)" % entropy(["y"] * 3 + ["n"] * 2))
    print("entropy(8 yes, 4 no)  = %.3f" % entropy(["y"] * 8 + ["n"] * 4))
    gain = information_gain([{"f": 0, "c": "y"}] * 0 + [{"f": "L", "c": "y"}] * 5 + [{"f": "L", "c": "n"}] * 2
                            + [{"f": "R", "c": "y"}] * 3 + [{"f": "R", "c": "n"}] * 2, "f", "c")
    print("information gain of the split of slide 25: %.4f" % gain)
    print("PlayTennis: entropy %.3f, Gini %.3f" % (entropy([r["PlayTennis"] for r in PLAY_ROWS]), gini([r["PlayTennis"] for r in PLAY_ROWS])))
    for f in NAMES[:-1]:
        print(f"  gain({f:9s}) = {information_gain(PLAY_ROWS, f, 'PlayTennis'):.3f}   gini gain = {information_gain(PLAY_ROWS, f, 'PlayTennis', gini):.3f}")
    tree = id3(PLAY_ROWS, NAMES[:-1])
    print(show(tree))
    for x in ({"Outlook": "Sunny", "Humidity": "High", "Temp": "Hot", "Wind": "Weak"},):
        print("X_test of slide 34 ->", id3_predict(tree, x))

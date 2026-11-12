# ci: run
"""Experiments of Chapter 6: agreement with scikit-learn, choosing k, scaling, dimensionality, NB and dependence."""
import numpy as np
from sklearn.datasets import load_wine, make_classification, make_moons
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import CategoricalNB as SkCatNB, GaussianNB as SkGNB, MultinomialNB as SkMNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import naive_bayes as nb
from knn import KNN, standardise

rng = np.random.default_rng(0)

# 1. Same answers as scikit-learn --------------------------------------------------------------
X = [r[:3] for r in nb.SHOPPING]; y = [r[3] for r in nb.SHOPPING]
enc = OrdinalEncoder().fit(X)
sk = SkCatNB(alpha=1e-10, force_alpha=True).fit(enc.transform(X), y)
q = ("Holiday", "Yes", "Yes")
ours = nb.CategoricalNB(alpha=0).fit(X, y).predict_proba(q)
theirs = dict(zip(sk.classes_, sk.predict_proba(enc.transform([q]))[0]))
assert all(abs(ours[c] - theirs[c]) < 1e-6 for c in ours)
print("shopping: our P(Yes) %.3f = sklearn %.3f" % (ours["Yes"], theirs["Yes"]))

docs = ["win money now", "win a prize now", "cheap money offer", "meeting at noon", "lunch at noon", "project meeting now"]
lab = ["spam", "spam", "spam", "ham", "ham", "ham"]
vocab = sorted({w for d in docs for w in d.split()})
count = lambda d: [d.split().count(w) for w in vocab]
skm = SkMNB(alpha=1.0).fit([count(d) for d in docs], lab)
mine = nb.MultinomialNB(alpha=1.0).fit(docs, lab).predict_proba("win meeting")
assert abs(mine["spam"] - skm.predict_proba([count("win meeting")])[0][list(skm.classes_).index("spam")]) < 1e-9
print("text: P(spam | 'win meeting') with alpha=1: %.3f (sklearn agrees)" % mine["spam"])

Xw, yw = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xw, yw, test_size=0.3, stratify=yw, random_state=0)
g1, g2 = nb.GaussianNB().fit(Xtr, ytr), SkGNB().fit(Xtr, ytr)
assert np.array_equal(g1.predict(Xte), g2.predict(Xte)) and np.allclose(g1.predict_proba(Xte), g2.predict_proba(Xte), atol=1e-6)
print("wine: GaussianNB test accuracy %.3f (identical to sklearn)" % np.mean(g1.predict(Xte) == yte))
Xs_tr, Xs_te = standardise(Xtr, Xte)
k1 = KNN(k=5).fit(Xs_tr, ytr); k2 = KNeighborsClassifier(5).fit(Xs_tr, ytr)
assert np.array_equal(k1.predict(Xs_te), k2.predict(Xs_te))
print("wine: our 5-NN (standardised) test accuracy %.3f (identical to sklearn)" % k1.score(Xs_te, yte))

# 2. Scaling: kNN on the wine data, raw against standardised features ----------------------------------------
cv = StratifiedKFold(5, shuffle=True, random_state=0)
raw = cross_val_score(KNeighborsClassifier(5), Xw, yw, cv=cv).mean()
std = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), Xw, yw, cv=cv).mean()
nbs = cross_val_score(SkGNB(), Xw, yw, cv=cv).mean()
print("wine 5-fold CV: 5-NN raw %.3f, 5-NN standardised %.3f, GaussianNB %.3f" % (raw, std, nbs))
print("wine feature standard deviations: min %.2f max %.1f" % (Xw.std(0).min(), Xw.std(0).max()))

# 3. Choosing k on a noisy two-moons problem -----------------------------------------------------------------
Xm, ym = make_moons(1000, noise=0.35, random_state=1)
Xa, Xb, ya, yb = train_test_split(Xm, ym, test_size=0.3, random_state=1, stratify=ym)
ks = list(range(1, 62, 2))
tr, cvs, te = [], [], []
for k in ks:
    m = KNeighborsClassifier(k).fit(Xa, ya)
    tr.append(1 - m.score(Xa, ya)); te.append(1 - m.score(Xb, yb))
    cvs.append(1 - cross_val_score(KNeighborsClassifier(k), Xa, ya, cv=cv).mean())
kbest = ks[int(np.argmin(cvs))]
print("moons: train error k=1 %.3f; best k by 5-fold CV = %d (CV error %.3f), test error there %.3f; k=1 test error %.3f"
      % (tr[0], kbest, min(cvs), te[ks.index(kbest)], te[0]))

# 4. Curse of dimensionality: relative contrast and accuracy ------------------------------------------------
print("relative contrast (Dmax - Dmin) / Dmin for 1000 uniform points in the unit cube:")
for d in (2, 10, 100, 1000):
    P = rng.random((1000, d)); qv = rng.random(d)
    dist = np.sqrt(((P - qv) ** 2).sum(1))
    print(f"  d={d:5d}: {(dist.max() - dist.min()) / dist.min():8.3f}")
for extra in (0, 10, 50, 200):
    Xn = np.hstack([Xm, rng.standard_normal((1000, extra))]) if extra else Xm
    a = cross_val_score(KNeighborsClassifier(15), Xn, ym, cv=cv).mean()
    b = cross_val_score(SkGNB(), Xn, ym, cv=cv).mean()
    print(f"  moons + {extra:3d} noise features: 15-NN accuracy {a:.3f}, GaussianNB {b:.3f}")

# 5. The independence assumption: copies of one feature make Naive Bayes over-confident ------------
Xc, yc = make_classification(3000, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8,
                             random_state=3, n_clusters_per_class=1, flip_y=0.05)
Xc1, Xc2, yc1, yc2 = train_test_split(Xc, yc, test_size=0.5, random_state=0)
for copies in (1, 3, 10):
    f = lambda A: np.hstack([A[:, [0]]] * copies + [A[:, [1]]])      # feature 0 repeated 'copies' times
    m = SkGNB().fit(f(Xc1), yc1)
    p = m.predict_proba(f(Xc2))
    conf = p.max(1)
    print(f"  {copies:2d} copies of x1: accuracy {m.score(f(Xc2), yc2):.3f}, mean confidence {conf.mean():.3f}, "
          f"log-loss {log_loss(yc2, np.clip(p, 1e-12, 1)):.3f}")

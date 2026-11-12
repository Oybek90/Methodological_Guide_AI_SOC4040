# ci: run
"""Experiments for Chapter 8: scratch vs scikit-learn, effect of C and gamma, scaling, kernels, SVR."""
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from svm import KernelSVM, LinearSVM, linear_kernel, poly_kernel, rbf_kernel

print("== 1. quadratic kernel check (slide 28): x=(1,2), z=(3,1) ==")
x, z = np.array([1.0, 2.0]), np.array([3.0, 1.0])
phi = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
print("K(x,z) = (x.z)^2 =", (x @ z) ** 2, "  phi(x).phi(z) =", round(float(phi(x) @ phi(z)), 6))

print("== 2. notebook data (standardised), linear SVM from scratch vs scikit-learn, same split ==")
X, y01 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, random_state=42)
Xtr0, Xte0, ytr01, yte01 = train_test_split(X, y01, test_size=0.2, random_state=50, stratify=y01)
sc = StandardScaler().fit(Xtr0)                       # statistics from the training set only
Xtr, Xte = sc.transform(Xtr0), sc.transform(Xte0)
ytr, yte = 2 * ytr01 - 1, 2 * yte01 - 1
sk = SVC(kernel="linear", C=1.0).fit(Xtr, ytr)
mine = LinearSVM(C=1.0, epochs=100, lr=0.05, seed=0).fit(Xtr, ytr)
obj = lambda w, b: 0.5 * w @ w + np.maximum(0, 1 - ytr * (Xtr @ w + b)).sum()
print("sklearn linear: w", np.round(sk.coef_[0], 3), "b", np.round(sk.intercept_[0], 3), "test acc %.3f" % sk.score(Xte, yte),
      "SVs", sk.n_support_.sum(), "objective %.2f" % obj(sk.coef_[0], sk.intercept_[0]))
print("scratch SGD   : w", np.round(mine.w_, 3), "b", round(mine.b_, 3), "test acc %.3f" % np.mean(mine.predict(Xte) == yte),
      "objective %.2f" % mine.objective(Xtr, ytr))
smo = KernelSVM(C=1.0, kernel=linear_kernel, seed=0).fit(Xtr, ytr)
print("scratch SMO (dual): test acc %.3f, SVs %d, max |decision value difference to sklearn| = %.4f" % (
    np.mean(smo.predict(Xte) == yte), smo.n_support_, np.abs(smo.decision_function(Xte) - sk.decision_function(Xte)).max()))
rbf = make_pipeline(StandardScaler(), SVC()).fit(Xtr0, ytr01)
rf1 = RandomForestClassifier(50, random_state=42).fit(Xtr0, ytr01)
print("single split (the notebook's): SVC rbf %.3f vs random forest %.3f" % (rbf.score(Xte0, yte01), rf1.score(Xte0, yte01)))

print("== 3. effect of C on a linear SVM (overlapping blobs, 200 points) ==")
Xb, yb = make_classification(200, 2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=1.0, random_state=3)
Xb = StandardScaler().fit_transform(Xb)
for C in (0.01, 0.1, 1, 10, 1000):
    m = SVC(kernel="linear", C=C).fit(Xb, yb)
    print(f"C={C:7} margin width 2/|w| = {2 / np.linalg.norm(m.coef_[0]):6.2f}  support vectors = {m.n_support_.sum():3d}  train acc {m.score(Xb, yb):.3f}")

print("== 4. kernels on two moons (n=400, noise 0.25), 5-fold CV accuracy ==")
Xm, ym = make_moons(400, noise=0.25, random_state=1)
for name, kw in (("linear", dict(kernel="linear")), ("poly deg 3", dict(kernel="poly", degree=3, coef0=1)),
                 ("rbf", dict(kernel="rbf"))):
    print(f"{name:11s} {cross_val_score(make_pipeline(StandardScaler(), SVC(C=1, **kw)), Xm, ym, cv=5).mean():.3f}")
ys = np.where(ym == 0, -1, 1)
a, b_, c, d = train_test_split(Xm, ys, test_size=0.3, random_state=0)
skr = SVC(kernel="rbf", C=1.0, gamma=1.0).fit(a, c)
mr = KernelSVM(C=1.0, kernel=rbf_kernel, gamma=1.0, seed=0).fit(a, c)
print("scratch SMO RBF vs sklearn: test acc %.3f vs %.3f, SVs %d vs %d, labels agree %.3f" % (
    np.mean(mr.predict(b_) == d), skr.score(b_, d), mr.n_support_, skr.n_support_.sum(), np.mean(mr.predict(b_) == skr.predict(b_))))

print("== 5. C and gamma (RBF) on moons: train vs test accuracy ==")
a, b_, c, d = train_test_split(Xm, ym, test_size=0.3, random_state=0)
for C, g in ((1, 0.01), (1, 1), (1, 100), (1000, 100), (0.01, 1)):
    m = SVC(C=C, gamma=g).fit(a, c)
    print(f"C={C:7} gamma={g:7} train {m.score(a, c):.3f} test {m.score(b_, d):.3f} SVs {m.n_support_.sum()}")
grid = GridSearchCV(SVC(), {"C": 10.0 ** np.arange(-2, 4), "gamma": 10.0 ** np.arange(-3, 3)}, cv=5).fit(a, c)
print("best", {k: float(v) for k, v in grid.best_params_.items()}, "cv %.3f, test %.3f" % (grid.best_score_, grid.score(b_, d)))

print("== 6. feature scaling, breast cancer, 10 stratified splits ==")
Xc, yc = load_breast_cancer(return_X_y=True)
r = []
for s in range(10):
    a, b_, c, d = train_test_split(Xc, yc, test_size=0.2, random_state=s, stratify=yc)
    r.append((SVC().fit(a, c).score(b_, d), make_pipeline(StandardScaler(), SVC()).fit(a, c).score(b_, d)))
r = np.array(r)
print("SVC raw %.3f +- %.3f | SVC scaled %.3f +- %.3f" % (r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 1].std()))

print("== 7. notebook comparison made fairer: SVC vs random forest, 10 splits, scaling inside the pipeline ==")
X, y01 = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
r = []
for s in range(10):
    a, b_, c, d = train_test_split(X, y01, test_size=0.2, random_state=s, stratify=y01)
    r.append((make_pipeline(StandardScaler(), SVC()).fit(a, c).score(b_, d),
              RandomForestClassifier(50, random_state=s).fit(a, c).score(b_, d)))
r = np.array(r)
print("SVC %.3f +- %.3f | random forest %.3f +- %.3f" % (r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 1].std()))

print("== 8. support vector regression: y = sin(x) + noise (80 points) ==")
rng = np.random.default_rng(0)
xs = np.sort(rng.uniform(0, 6, 80))[:, None]
ys = np.sin(xs[:, 0]) + rng.normal(0, 0.2, 80)
for eps in (0.01, 0.1, 0.3, 0.6):
    m = SVR(C=10, gamma=1.0, epsilon=eps).fit(xs, ys)
    xg = np.linspace(0, 6, 300)[:, None]
    print(f"epsilon={eps:4} support vectors {len(m.support_):3d} of 80   MSE vs true sin {np.mean((m.predict(xg) - np.sin(xg[:, 0])) ** 2):.4f}")

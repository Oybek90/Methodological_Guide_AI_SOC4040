# ci: run
"""Extra PCA experiments: why scaling matters, leakage, choosing k, whitening."""
import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Which columns of make_classification carry the variance? (shuffle=False
#    keeps the order: informative, redundant, noise)
X, y = make_classification(n_samples=1000, n_features=800, n_informative=10,
                           n_redundant=5, shuffle=False, random_state=0)
v = X.var(axis=0, ddof=1)
print("var informative (10)   : mean %.2f  max %.2f" % (v[:10].mean(), v[:10].max()))
print("var redundant   (5)    : mean %.2f" % v[10:15].mean())
print("var noise       (785)  : mean %.3f min %.3f max %.3f" % (v[15:].mean(), v[15:].min(), v[15:].max()))
lam = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
print("top-15 eigenvalues      :", np.round(lam[:15], 1))
print("eigenvalues 16..20      :", np.round(lam[15:20], 2))
Xs = StandardScaler().fit_transform(X)
lam_s = np.linalg.eigvalsh(np.cov(Xs, rowvar=False))[::-1]
print("standardised top-5      :", np.round(lam_s[:5], 2), " eigenvalue 16:", round(lam_s[15], 2))
print("Marchenko-Pastur upper edge (1+sqrt(n/m))^2 for pure noise:", round((1 + np.sqrt(800 / 1000)) ** 2, 3))

# 2. Scaling demo of the exercise: x2 = 100*x1 + N(0,25) (sd 5)
rng = np.random.default_rng(0)
x1 = rng.normal(0, 1, 500)
x2 = 100 * x1 + rng.normal(0, 5, 500)
D = np.c_[x1, x2]
for name, M in (("raw", D), ("standardised", StandardScaler().fit_transform(D))):
    p = PCA(2).fit(M)
    print(f"{name:13s} PC1 = {np.round(p.components_[0], 4)}  EVR = {np.round(p.explained_variance_ratio_, 4)}")


# 3. Leakage: PCA fitted on all data vs inside the training split
def leak(seed, k=50):
    X, y = make_classification(n_samples=1000, n_features=800, n_informative=10,
                               n_redundant=5, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    clean = make_pipeline(PCA(k, random_state=seed), SVC(kernel="linear", random_state=seed)).fit(Xtr, ytr).score(Xte, yte)
    Z = PCA(k, random_state=seed).fit_transform(X)                  # WRONG: sees test rows
    Ztr, Zte, ytr2, yte2 = train_test_split(Z, y, test_size=0.2, random_state=seed)
    dirty = SVC(kernel="linear", random_state=seed).fit(Ztr, ytr2).score(Zte, yte2)
    return clean, dirty


r = np.array([leak(s) for s in range(10)])
d = r[:, 1] - r[:, 0]
print("leakage: clean %.3f  leaky %.3f  diff mean %+.3f sd %.3f  leaky>clean in %d/10"
      % (r[:, 0].mean(), r[:, 1].mean(), d.mean(), d.std(ddof=1), (d > 0).sum()))

# 4. Choosing k on digits
Xd, yd = load_digits(return_X_y=True)
cum = np.cumsum(PCA(svd_solver="full").fit(Xd).explained_variance_ratio_)
k80 = int(np.searchsorted(cum, 0.80) + 1)
print("k for 80%% variance: %d" % k80)
lr = lambda: LogisticRegression(max_iter=5000)
base = cross_val_score(lr(), Xd, yd, cv=5).mean()
print("LR on 64 pixels           : %.4f" % base)
for k in (k80, 10, 20, 30, 40):
    s = cross_val_score(make_pipeline(PCA(k), lr()), Xd, yd, cv=5).mean()
    print(f"PCA({k:2d}) + LR            : {s:.4f}")

# 5. Whitening: unit variance, zero correlation
p = PCA(n_components=5, whiten=True).fit(Xd)
Z = p.transform(Xd)
print("whitened variances:", np.round(Z.var(axis=0, ddof=1), 3))
print("max |offdiag corr| :", np.abs(np.corrcoef(Z.T) - np.eye(5)).max().round(12))

# 6. Arbitrary axis signs: a linear SVM does not care, a random forest does
from sklearn.ensemble import RandomForestClassifier
from pca_scratch import PCAScratch

X, y = make_classification(n_samples=1000, n_features=800, n_informative=10,
                           n_redundant=5, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
ps = PCAScratch(50).fit(Xtr)
Z, Zt = ps.transform(Xtr), ps.transform(Xte)
sk = PCA(50, random_state=42).fit(Xtr)
cos = np.abs((ps.components_ * sk.components_).sum(axis=1))
print("scratch vs sklearn |cos|: first 10 min %.6f, axes 11-50 min %.3f" % (cos[:10].min(), cos[10:].min()))
rng = np.random.default_rng(0)
rf, sv = [], []
for _ in range(20):
    s = rng.choice([-1.0, 1.0], size=50)
    rf.append(RandomForestClassifier(random_state=42).fit(Z * s, ytr).score(Zt * s, yte))
    sv.append(SVC(kernel="linear").fit(Z * s, ytr).score(Zt * s, yte))
print("20 random sign patterns: RF min %.3f max %.3f mean %.3f ; SVM values %s"
      % (min(rf), max(rf), np.mean(rf), sorted(set(np.round(sv, 3)))))
print("binomial s.e. of an accuracy of 0.8 on 200 test points: %.3f" % np.sqrt(0.8 * 0.2 / 200))

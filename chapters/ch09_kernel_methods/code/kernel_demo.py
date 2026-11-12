# ci: run
"""Verify the numbers of the lecture: dimension table, 121 = 121, Gram properties, closure rules, tuning, kernel comparison."""
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import KernelPCA
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from kernels import (KernelSVM, laplacian_kernel, linear_kernel, phi_quadratic, phi_quadratic_bias, is_psd,
                     poly_feature_dim, poly_kernel, rbf_kernel, sigmoid_kernel)

# 1. dimension table (slide 13) and memory estimates (slides 13 and 19)
for n, d in [(2, 2), (100, 2), (100, 5), (784, 3), (784, 5)]:
    D = poly_feature_dim(n, d)
    print(f"n={n:4d} d={d}  D={D:>16,d}  float64 storage of one phi(x): {8 * D / 1e9:,.4g} GB")
print("Gram matrix for n=100000:", f"{1e5 ** 2 * 8 / 1e9:.0f} GB")

# 2. kernel trick by hand (slides 16-17) and the 6-feature map of quiz question 1
x, z = np.array([1.0, 2.0]), np.array([3.0, 4.0])
assert np.isclose(phi_quadratic(x) @ phi_quadratic(z), 121.0) and (x @ z) ** 2 == 121.0
print("quadratic:", phi_quadratic(x) @ phi_quadratic(z), (x @ z) ** 2)
print("(x.z+1)^2:", phi_quadratic_bias(x) @ phi_quadratic_bias(z), (x @ z + 1) ** 2)

# 3. Gram matrix properties and closure rules (slides 19-20)
rng = np.random.default_rng(0)
P = rng.normal(size=(40, 3))
K1, K2 = rbf_kernel(P, P, 0.5), poly_kernel(P, P, 2, 1.0, 1.0)
print("RBF diag all 1:", bool(np.allclose(np.diag(K1), 1)), " PSD:", is_psd(K1), is_psd(K2), is_psd(laplacian_kernel(P, P, 0.5)))
f = rng.normal(size=40)
rules = {"K1+K2": K1 + K2, "3*K1": 3 * K1, "K1*K2 (Hadamard)": K1 * K2, "f(x)K1f(z)": f[:, None] * K1 * f[None, :], "exp(K1)": np.exp(K1)}
for name, K in rules.items():
    print(f"  {name:18s} PSD: {is_psd(K)}")
print("  K1-K2 (not a rule)  PSD:", is_psd(K1 - K2), " min eig:", round(float(np.linalg.eigvalsh(K1 - K2).min()), 3))
for g, r in [(1.0, 0.0), (0.5, -1.0), (0.1, -1.0)]:
    Ks = sigmoid_kernel(P, P, g, r)
    print(f"  sigmoid gamma={g} r={r}: PSD {is_psd(Ks)}  min eig {np.linalg.eigvalsh(Ks).min():.3f}")

# 4. datasets of the lecture: linear vs polynomial vs RBF vs sigmoid
def xor(n=300, seed=0):
    r = np.random.default_rng(seed); X = r.uniform(-1, 1, (n, 2)); return X, (X[:, 0] * X[:, 1] > 0).astype(int)

data = {"XOR": xor(), "circles": make_circles(300, noise=0.1, factor=0.4, random_state=0), "moons": make_moons(300, noise=0.2, random_state=0)}
print("\nheld-out accuracy (30 % test, stratified, standardised):")
cfg = {"linear": dict(kernel="linear"), "poly d=2": dict(kernel="poly", degree=2, coef0=1, gamma=1), "poly d=5": dict(kernel="poly", degree=5, coef0=1, gamma=1),
       "rbf": dict(kernel="rbf"), "sigmoid": dict(kernel="sigmoid")}
for name, (X, y) in data.items():
    a, b, c, e = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    print(f"{name:8s}", {k: round(make_pipeline(StandardScaler(), SVC(**v)).fit(a, c).score(b, e), 3) for k, v in cfg.items()})

# 5. the grid search of slide 30 and gamma sweep of slide 25
X, y = make_moons(200, noise=0.12, random_state=2)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
gs = GridSearchCV(pipe, {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.01, 0.1, 1, 10]}, cv=5).fit(Xtr, ytr)
print("\nGridSearchCV:", gs.best_params_, round(gs.best_score_, 3), round(gs.score(Xte, yte), 3))
for g in (0.05, 1, 50):
    m = make_pipeline(StandardScaler(), SVC(gamma=g)).fit(Xtr, ytr)
    print(f"gamma={g:5}: train {m.score(Xtr, ytr):.3f} test {m.score(Xte, yte):.3f} SVs {m[-1].n_support_.sum()}")

# 6. our solver against scikit-learn (the library uses a separate bias, so decisions differ slightly)
sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xte)
ys, yt = 2 * ytr - 1, 2 * yte - 1
ours = KernelSVM(rbf_kernel, C=1.0, gamma=1.0).fit(A, ys)
sk = SVC(C=1.0, gamma=1.0).fit(A, ys)
print(f"\nours {ours.score(B, yt):.3f} (SV {len(ours.alpha_)}, epochs {ours.n_epochs_}) | sklearn {sk.score(B, yt):.3f} (SV {sk.n_support_.sum()}); agreement {np.mean(ours.predict(B) == sk.predict(B)):.3f}")

# 7. large n: exact kernel SVC against Nystroem / random Fourier features + linear SVM
Xl, yl = make_moons(6000, noise=0.25, random_state=0)
a, b, c, e = train_test_split(Xl, yl, test_size=0.3, stratify=yl, random_state=0)
import time
for name, model in {"exact SVC": SVC(gamma=2.0), "Nystroem(100)+LinearSVC": make_pipeline(Nystroem(gamma=2.0, n_components=100, random_state=0), LinearSVC(random_state=0)),
                    "RFF(100)+LinearSVC": make_pipeline(RBFSampler(gamma=2.0, n_components=100, random_state=0), LinearSVC(random_state=0)), "LinearSVC": LinearSVC(random_state=0)}.items():
    t = time.perf_counter(); model.fit(a, c); dt = time.perf_counter() - t
    print(f"{name:26s} test {model.score(b, e):.3f}")

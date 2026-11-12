# ci: run
"""Check the scratch implementation against scikit-learn and against the SVD."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from pca_scratch import PCAScratch

X, _ = load_digits(return_X_y=True)          # 1797 x 64
k = 10
mine = PCAScratch(k).fit(X)
ref = PCA(n_components=k, svd_solver="full").fit(X)

# Eigenvalues agree
print("max |lambda diff|        :", np.abs(mine.eigenvalues_[:k] - ref.explained_variance_).max())

# Directions agree up to sign (each PC is defined only up to +/-1)
cos = np.abs(np.sum(mine.components_ * ref.components_, axis=1))
print("min |cos| of PC pairs    :", cos.min())

# SVD relation: lambda_i = sigma_i^2 / (m-1)
Xc = X - X.mean(axis=0)
sigma = np.linalg.svd(Xc, compute_uv=False)
print("max |sigma^2/(m-1)-lam|  :", np.abs(sigma[:k] ** 2 / (len(X) - 1) - ref.explained_variance_).max())

# Float n_components = target variance fraction
p95 = PCA(n_components=0.95, svd_solver="full").fit(X)
print("components for 95% var   :", p95.n_components_)

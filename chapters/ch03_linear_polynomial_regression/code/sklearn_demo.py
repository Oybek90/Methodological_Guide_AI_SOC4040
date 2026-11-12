# ci: run
"""scikit-learn pipelines on the diabetes data: linear, degree-2 polynomial, polynomial + ridge."""
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)


def poly(estimator):
    # scale -> degree-2 features -> scale again -> regression; the scalers learn from the training set only
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), StandardScaler(), estimator)


models = {
    "linear": make_pipeline(StandardScaler(), LinearRegression()),
    "degree 2": poly(LinearRegression()),
    "degree 2 + ridge(10)": poly(Ridge(alpha=10)),
    "degree 2 + ridge(100)": poly(Ridge(alpha=100)),
}
print(f"{'model':24s} {'train R2':>9s} {'test R2':>9s}   (n_train={len(ytr)}, n_test={len(yte)})")
for name, mdl in models.items():
    mdl.fit(Xtr, ytr)
    print(f"{name:24s} {mdl.score(Xtr, ytr):9.3f} {mdl.score(Xte, yte):9.3f}")

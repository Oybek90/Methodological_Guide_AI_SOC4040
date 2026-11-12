# ci: run
"""Does PCA help a classifier on wide, noisy data? Honest multi-seed test."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run(seed, k=50, scale=False):
    X, y = make_classification(n_samples=1000, n_features=800, n_informative=10,
                               n_redundant=5, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    out = {}
    models = {"SVM": lambda: SVC(kernel="linear", random_state=seed),
              "RF": lambda: RandomForestClassifier(n_estimators=100, random_state=seed)}
    for name, make in models.items():
        out[name + "-raw"] = make().fit(Xtr, ytr).score(Xte, yte)
        # PCA lives INSIDE the pipeline -> fitted on the training split only
        steps = ([StandardScaler()] if scale else []) + [PCA(n_components=k, random_state=seed), make()]
        pipe = make_pipeline(*steps)
        out[name + "-pca"] = pipe.fit(Xtr, ytr).score(Xte, yte)
    return out


if __name__ == "__main__":
    for scale in (False, True):
        rows = [run(s, scale=scale) for s in range(10)]
        print(f"--- StandardScaler before PCA: {scale} ---")
        print("seed 0:", {k: round(v, 3) for k, v in rows[0].items()})
        for key in rows[0]:
            v = np.array([r[key] for r in rows])
            print(f"{key:8s} mean={v.mean():.3f} std={v.std(ddof=1):.3f}")
        for m in ("SVM", "RF"):
            d = np.array([r[m + "-pca"] - r[m + "-raw"] for r in rows])
            print(f"{m} gain: mean {d.mean():+.3f}, wins {(d > 0).sum()}/10")
    print("seed 42 (slide setting, no scaling):",
          {k: round(v, 3) for k, v in run(42).items()})

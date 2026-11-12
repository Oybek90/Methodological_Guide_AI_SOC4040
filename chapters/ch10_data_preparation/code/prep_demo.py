# ci: run
"""Numbers quoted in Chapter 10: outlier rules, accuracy paradox, encoders, and four kinds of leakage (seeded)."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

from prep_utils import iqr_filter_rows, iqr_outliers, smote, target_encode, zscore_outliers

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
T = pd.read_csv("titanic.csv")

# 1. the IQR snippet of slide 13 does not remove rows
num = T[["age", "fare", "sibsp", "parch"]]
q1, q3 = num.quantile(0.25), num.quantile(0.75); iqr = q3 - q1
slide = num[(num >= q1 - 1.5 * iqr) & (num <= q3 + 1.5 * iqr)]
print(f"slide IQR code: {slide.shape[0]} rows kept of {len(num)}, but {int(slide.isna().sum().sum() - num.isna().sum().sum())} extra NaN cells created")
print("rows inside the fences in every column (.all(axis=1)):", len(iqr_filter_rows(num)), "of", len(num), "| fare outliers:", int(iqr_outliers(T["fare"]).sum()), "| z>3:", int(zscore_outliers(T["fare"]).sum()))

# 2. with the population standard deviation (scipy.stats.zscore default) |z| <= sqrt(n-1): a cut-off of 3 cannot fire for n <= 10
for n in (5, 10, 11, 20):
    x = np.r_[np.zeros(n - 1), 1.0]
    print(f"n={n:2d}: largest possible |z| = {np.abs((x - x.mean()) / x.std()).max():.3f}  (bound sqrt(n-1) = {np.sqrt(n - 1):.3f})")

# 3. accuracy paradox of slide 24
y = np.r_[np.zeros(9980), np.ones(20)]; p = np.zeros(10000)
print(f"always 'normal': accuracy {accuracy_score(y, p):.3f}, recall {recall_score(y, p):.1f}")

# 4. LabelEncoder is for targets and sorts alphabetically; OrdinalEncoder takes the real order
sizes = np.array(["small", "medium", "large"])
print("LabelEncoder  :", dict(zip(sizes, LabelEncoder().fit_transform(sizes))))
print("OrdinalEncoder:", dict(zip(sizes, OrdinalEncoder(categories=[["small", "medium", "large"]]).fit_transform(sizes[:, None]).ravel())))

# 5. get_dummies before the split gives different columns; OneHotEncoder(handle_unknown) is safe
tr, te = pd.DataFrame({"c": ["a", "b", "a"]}), pd.DataFrame({"c": ["a", "z"]})
print("get_dummies columns train/test:", list(pd.get_dummies(tr).columns), list(pd.get_dummies(te).columns))
oh = OneHotEncoder(handle_unknown="ignore").fit(tr)
print("OneHotEncoder on test:", oh.transform(te).toarray().tolist())

# 6. Ridge does not select features, Lasso does
Xr = rng.normal(size=(200, 20)); yr = Xr[:, 0] * 3 + Xr[:, 1] * 2 + rng.normal(size=200)
print("zero coefficients: Ridge", int((Ridge(alpha=10).fit(Xr, yr).coef_ == 0).sum()), " Lasso", int((Lasso(alpha=0.3).fit(Xr, yr).coef_ == 0).sum()), "of 20")

# 7. leakage I: feature selection on all data, labels are pure noise
Xn = rng.normal(size=(100, 1000)); yn = rng.integers(0, 2, 100)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
leak = cross_val_score(LogisticRegression(max_iter=1000), SelectKBest(f_classif, k=20).fit_transform(Xn, yn), yn, cv=cv).mean()
ok = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000)), Xn, yn, cv=cv).mean()
print(f"noise labels, selection outside CV: {leak:.3f}   inside the pipeline: {ok:.3f}")

# 8. leakage II: SMOTE before the split
Xi, yi = make_classification(1000, 10, n_informative=3, weights=[0.92], flip_y=0.05, class_sep=0.8, random_state=1)
Xa, ya = smote(Xi, yi, seed=0)
Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, stratify=ya, random_state=0)
m = RandomForestClassifier(200, random_state=0).fit(Xtr, ytr)
print(f"SMOTE then split : minority F1 {f1_score(yte, m.predict(Xte)):.3f}")
Xtr, Xte, ytr, yte = train_test_split(Xi, yi, test_size=0.3, stratify=yi, random_state=0)
Xs, ys = smote(Xtr, ytr, seed=0)
m = RandomForestClassifier(200, random_state=0).fit(Xs, ys)
print(f"split then SMOTE : minority F1 {f1_score(yte, m.predict(Xte)):.3f}   (no resampling: {f1_score(yte, RandomForestClassifier(200, random_state=0).fit(Xtr, ytr).predict(Xte)):.3f})")

# 9. leakage III: naive target encoding of a 100-level random category
cat = rng.integers(0, 100, 600).astype(str); yt = rng.integers(0, 2, 600)
naive = pd.Series(yt).groupby(cat).transform("mean").to_numpy()            # uses each row's own label
a, b, c, d = train_test_split(naive, yt, test_size=0.3, random_state=0)
print(f"naive target encoding  : accuracy {LogisticRegression().fit(a[:, None], c).score(b[:, None], d):.3f}")
ctr, cte, ytr2, yte2 = train_test_split(cat, yt, test_size=0.3, random_state=0)
etr, ete = target_encode(ctr, ytr2, cte)
print(f"out-of-fold smoothed   : accuracy {LogisticRegression().fit(etr[:, None], ytr2).score(ete[:, None], yte2):.3f}")

# 10. leakage IV: a scaler fitted on everything (usually a small effect)
Xs_ = rng.lognormal(size=(300, 5)); ys_ = (Xs_[:, 0] > np.median(Xs_[:, 0])).astype(int)
from sklearn.neighbors import KNeighborsClassifier
cv2 = StratifiedKFold(5, shuffle=True, random_state=0)
l1 = cross_val_score(KNeighborsClassifier(), StandardScaler().fit_transform(Xs_), ys_, cv=cv2).mean()
l2 = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier()), Xs_, ys_, cv=cv2).mean()
print(f"scaler outside CV {l1:.3f} | inside {l2:.3f}")

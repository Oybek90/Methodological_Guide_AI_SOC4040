# ci: run
"""Leak-free, stage-by-stage preparation of the Titanic data with repeated stratified cross-validation."""
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

warnings.filterwarnings("ignore")
T = pd.read_csv("titanic.csv")
y = T["survived"].to_numpy()
BASE = ["pclass", "age", "sibsp", "parch", "fare"]


def engineer(X):
    """Row-wise features: nothing is learned from the data, so this step cannot leak."""
    X = X.copy()
    X["family_size"] = X["sibsp"] + X["parch"] + 1
    X["is_alone"] = (X["family_size"] == 1).astype(int)
    X["fare_per_person"] = X["fare"] / X["family_size"]
    X["age_missing"] = X["age"].isna().astype(int)
    return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise to Q1 - k*IQR .. Q3 + k*IQR, with the fences learned on the training fold only."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.lo_, self.hi_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)


def make(stage, model="rf"):
    num = ["pclass", "age", "sibsp", "parch", "fare"]
    cat = []
    pre_steps = []
    if stage >= 5:
        pre_steps.append(("fe", FunctionTransformer(engineer)))
        num = num + ["family_size", "is_alone", "fare_per_person", "age_missing"]
    imputer = SimpleImputer(strategy="median") if stage == 0 else KNNImputer(n_neighbors=5)
    num_steps = [("imp", imputer)]
    if stage >= 4:
        num_steps.append(("clip", IQRClipper()))
    if stage >= 3:
        num_steps.append(("scale", StandardScaler() if model == "lr" else RobustScaler()))
    transformers = [("num", Pipeline(num_steps), num)]
    if stage >= 2:
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("oh", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("cat", cat_pipe, ["sex", "embarked"]))
    steps = pre_steps + [("ct", ColumnTransformer(transformers))]
    if stage >= 6:
        steps.append(("sel", SelectFromModel(RandomForestClassifier(200, random_state=0), max_features=8, threshold=-np.inf)))
    clf = RandomForestClassifier(300, random_state=0, min_samples_leaf=2) if model == "rf" else LogisticRegression(max_iter=2000)
    if stage >= 7:
        return ImbPipeline(steps + [("smote", SMOTE(random_state=0)), ("clf", clf)])
    return Pipeline(steps + [("clf", clf)])


NAMES = ["0 raw numeric, median", "1 + KNN imputation", "2 + one-hot sex, embarked", "3 + scaling", "4 + IQR clipping",
         "5 + feature engineering", "6 + feature selection (8)", "7 + SMOTE"]
if __name__ == "__main__":
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    print(f"{'stage':28s} {'RF acc':>14s} {'LR acc':>14s} {'RF F1(surv)':>14s}")
    for s, name in enumerate(NAMES):
        out = {}
        for mdl in ("rf", "lr"):
            r = cross_validate(make(s, mdl), T[cols], y, cv=cv, scoring=["accuracy", "f1"], n_jobs=-1)
            out[mdl] = (r["test_accuracy"], r["test_f1"])
        a, b = out["rf"][0], out["lr"][0]
        print(f"{name:28s} {a.mean():.3f}+-{a.std():.3f}  {b.mean():.3f}+-{b.std():.3f}  {out['rf'][1].mean():.3f}+-{out['rf'][1].std():.3f}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if __package__:
    from .preprocessing import build_preprocessor
else:
    from preprocessing import build_preprocessor


def build_logistic_model(X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X, "logistic")),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def build_random_forest_model(X, n_jobs=2):
    return Pipeline([
        ("preprocessor", build_preprocessor(X, "tree")),
        ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=n_jobs)),
    ])

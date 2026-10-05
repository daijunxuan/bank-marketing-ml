from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    if set(numeric + categorical) != set(X.columns):
        raise ValueError("Unsupported feature dtype; use numeric or categorical columns")
    return numeric, categorical


def build_preprocessor(X, model_type="logistic"):
    numeric, categorical = get_feature_types(X)
    if model_type not in {"logistic", "tree"}:
        raise ValueError(f"Unknown model type: {model_type}")
    return ColumnTransformer([
        ("num", StandardScaler() if model_type == "logistic" else "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

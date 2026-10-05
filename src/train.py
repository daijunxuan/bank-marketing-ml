import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

if __package__:
    from .models import build_logistic_model, build_random_forest_model
    from .evaluate import evaluate_model, find_best_threshold
else:
    from models import build_logistic_model, build_random_forest_model
    from evaluate import evaluate_model, find_best_threshold

ROOT = Path(__file__).resolve().parents[1]


def prepare_data(data_path):
    df = pd.read_csv(data_path, sep=";")
    if not {"y", "duration"}.issubset(df.columns):
        raise ValueError("Expected bank-additional-full.csv with y and duration columns")
    if not df["y"].isin(["yes", "no"]).all():
        raise ValueError("Target y must contain only yes/no values")
    X = df.drop(columns=["y", "duration"])
    y = df["y"].map({"no": 0, "yes": 1})
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train(data_path=ROOT / "data/bank-additional-full.csv",
          model_path=ROOT / "models/model.joblib", results_path=ROOT / "results/metrics.json",
          n_jobs=2):
    data_path, model_path, results_path = map(Path, (data_path, model_path, results_path))
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing {data_path}. Run: python scripts/download_data.py")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path)
    candidates = {
        "Logistic Regression": build_logistic_model(X_train),
        "Random Forest": build_random_forest_model(X_train, n_jobs),
    }
    validation, thresholds = {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_val)[:, 1]
        threshold = float(find_best_threshold(y_val, probabilities)["threshold"])
        thresholds[name] = threshold
        validation[name] = evaluate_model(y_val, probabilities, threshold)
        validation[name]["threshold"] = threshold
    # Select only using validation data; test the selected model exactly once.
    selected = max(validation, key=lambda name: validation[name]["PR-AUC"])
    model = candidates[selected]
    threshold = thresholds[selected]
    test = evaluate_model(y_test, model.predict_proba(X_test)[:, 1], threshold)
    versions = {name: importlib.metadata.version(name)
                for name in ("pandas", "numpy", "scikit-learn", "scipy", "joblib")}
    results = {
        "data_file": data_path.name,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "split": {"train": len(y_train), "validation": len(y_val), "test": len(y_test),
                  "random_state": 42, "stratified": True},
        "excluded_features": ["duration"],
        "selection_metric": "validation average precision (labelled PR-AUC for compatibility)",
        "threshold_selection": "maximum validation F1 for each candidate",
        "validation": validation, "selected_model": selected,
        "threshold": threshold, "test": test,
        "test_positive_rate": float(y_test.mean()), "packages": versions,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "feature_columns": X_train.columns.tolist(),
                 "metadata": results}, model_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare models and save the selected model with its threshold")
    parser.add_argument("--data", type=Path, default=ROOT / "data/bank-additional-full.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models/model.joblib")
    parser.add_argument("--results", type=Path, default=ROOT / "results/metrics.json")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    try:
        train(args.data, args.model, args.results, args.n_jobs)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

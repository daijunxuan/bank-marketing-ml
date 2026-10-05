import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def predict(data_path, model_path, output_path):
    artifact = joblib.load(model_path)
    data = pd.read_csv(data_path, sep=";")
    missing = set(artifact["feature_columns"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    probabilities = artifact["model"].predict_proba(data[artifact["feature_columns"]])[:, 1]
    result = pd.DataFrame({"probability": probabilities,
                           "prediction": (probabilities >= artifact["threshold"]).astype(int)})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict using the saved validation threshold")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/model.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "results/predictions.csv")
    args = parser.parse_args()
    try:
        predict(args.input, args.model, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

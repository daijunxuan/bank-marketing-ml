import io
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from scripts.download_data import csv_from_archive
from src.predict import predict
from src.train import prepare_data, train


def dataset(tmp_path):
    rng = np.random.default_rng(42)
    age = rng.integers(18, 80, 160)
    frame = pd.DataFrame({"age": age, "job": pd.Series(np.where(age > 45, "admin", "student"), dtype="string"),
                          "duration": rng.integers(1, 400, 160), "y": np.where(age > 45, "yes", "no")})
    path = tmp_path / "input.csv"
    frame.to_csv(path, sep=";", index=False)
    return path, frame


def test_saved_artifact_preserves_threshold_and_predicts_unknown_category(tmp_path):
    path, frame = dataset(tmp_path)
    model = tmp_path / "nested/model.joblib"
    result = train(path, model, tmp_path / "results/metrics.json", n_jobs=1)
    artifact = joblib.load(model)
    assert "duration" not in artifact["feature_columns"]
    assert artifact["threshold"] == result["threshold"]
    frame.loc[0, "job"] = "never-seen-before"
    frame.to_csv(path, sep=";", index=False)
    predictions = predict(path, model, tmp_path / "out/predictions.csv")
    np.testing.assert_array_equal(predictions.prediction,
                                  (predictions.probability >= artifact["threshold"]).astype(int))


def test_stratified_splits_do_not_overlap(tmp_path):
    path, _ = dataset(tmp_path)
    a, b, c, *_ = prepare_data(path)
    assert not set(a.index) & set(b.index)
    assert not set(a.index) & set(c.index)
    assert not set(b.index) & set(c.index)


def test_download_supports_nested_uci_archive():
    inner = io.BytesIO()
    with ZipFile(inner, "w") as z:
        z.writestr("bank-additional/bank-additional-full.csv", "age;y\n40;yes\n")
    outer = io.BytesIO()
    with ZipFile(outer, "w") as z:
        z.writestr("bank-additional.zip", inner.getvalue())
    assert csv_from_archive(outer.getvalue()) == b"age;y\n40;yes\n"

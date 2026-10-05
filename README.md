# Bank Marketing Prediction

[![Tests](https://github.com/daijunxuan/bank-marketing-ml/actions/workflows/test.yml/badge.svg)](https://github.com/daijunxuan/bank-marketing-ml/actions/workflows/test.yml)

A reproducible classification baseline for predicting term-deposit subscription using the UCI Bank Marketing **bank-additional-full** variant. The workflow compares logistic regression and random forest, removes the post-call `duration` feature, selects a threshold using validation data, and saves the model together with that threshold.

## Quick start

Python 3.11–3.13; dependencies are pinned:

```bash
git clone https://github.com/daijunxuan/bank-marketing-ml.git
cd bank-marketing-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/download_data.py
python -m src.train
python -m src.predict --input data/bank-additional-full.csv --output results/predictions.csv
python -m pytest -q
```

The download script retrieves the official UCI archive, reads the nested `bank-additional.zip`, verifies the CSV's SHA256 and writes `data/bank-additional-full.csv`. It never substitutes the older `bank-full.csv` variant. Existing data is preserved unless `--force` is passed. Model and output directories are created automatically. `python src/train.py` also remains supported.

Training options: `--data PATH`, `--model PATH`, `--results PATH`, and `--n-jobs N`. Defaults resolve relative to the repository, so running the script by its absolute path from another directory also works. Explicit relative paths resolve from the current directory.

Outputs:

- `models/model.joblib`: model, validation threshold, ordered feature names and metadata.
- `results/metrics.json`: split sizes, validation/test metrics, data hash and package versions.
- `results/predictions.csv`: probabilities and labels calculated with the saved threshold.

Generated models and data are ignored. [The committed reference run](results/reference.json) records a full real-data run; it is separate from each local run's output.

## Dataset and evaluation

[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306), contains the selected 41,188-row variant with 20 input variables before removing `duration`. Target `y`: no → 0, yes → 1. Dataset credit: S. Moro, P. Rita and P. Cortez; the data is CC BY 4.0. No new code license is asserted for this repository.

Split: 28,831 training / 6,178 validation / 6,179 test rows, stratified with seed 42. Scaling and one-hot encoding are fitted only on training data; unseen categories are supported. The literal `unknown` category is retained. `duration` is unavailable before a call ends and is excluded from both candidates.

Each candidate's threshold maximizes validation F1. Validation **average precision** selects the model. Only the selected model is evaluated on the held-out test, without refitting on validation. In the original metric dictionary this quantity is labelled `PR-AUC`; it is sklearn `average_precision_score`, not trapezoidal PR-curve area.

## Reproduced results

These tables were generated from the committed reference run with the pinned environment and input SHA256 `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`.

| Validation model | ROC-AUC | Average precision | F1 at selected threshold | Threshold |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.802 | 0.470 | 0.495 | 0.2208 |
| Random Forest | 0.783 | 0.441 | 0.475 | 0.2800 |

Selected model: **Logistic Regression**. Saved threshold: **0.2208**.

| Held-out test metric | Score |
|---|---:|
| ROC-AUC | 0.805 |
| PR-AUC | 0.468 |
| Precision | 0.467 |
| Recall | 0.560 |
| F1 | 0.509 |

Test positive rate: 0.113; this is also the constant-score average-precision baseline. Run the training command to generate your own complete, unrounded results.

## Notebooks

1. `01_exploration.ipynb`: target imbalance, preprocessing and an explicit with/without-duration leakage comparison. Early exploratory models intentionally contain `duration`; they are not the final model.
2. `02_model_comparison.ipynb`: invokes the same training workflow, displays both validation candidates and the selected held-out test result, and verifies the saved threshold.

Both notebooks have been executed with the documented data and dependencies. Download the data before using Run All.

## Limits and next experiments

The dataset is ordered by time, while this baseline uses a random split. A chronological holdout is needed before making deployment/generalization claims. F1 threshold selection does not encode campaign costs or calibrated business value. This remains an offline baseline; production serving, monitoring and drift handling are not implemented. Segment-level error analysis should be supported by measured tables rather than inferred customer stories.

Tests verify non-overlapping splits, exclusion of `duration`, a portable model/threshold artifact, prediction with unseen categories and parsing of the nested UCI archive. CI uses small synthetic fixtures; it does not download or retrain the full dataset on every push.

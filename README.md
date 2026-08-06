# Bank Marketing Prediction

An end-to-end machine learning project for predicting whether a customer will subscribe to a bank term deposit.

The project focuses on building a production-oriented ML pipeline, including preprocessing, model comparison, threshold optimization, leakage analysis, and error analysis.

---

## Problem

Bank marketing campaigns aim to identify customers who are likely to subscribe to term deposits.

Given customer demographic information, campaign history, and economic indicators, the goal is to predict subscription probability.

This is a binary classification problem:

- 0: No subscription
- 1: Subscription

---

## Dataset

Dataset:

UCI Bank Marketing Dataset

Features:

- Demographic information
- Campaign contact history
- Economic indicators

Target:

`y`

where:

- no → 0
- yes → 1

---

## Project Structure

```text
bank-marketing-ml/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   └── train.py
├── models/
└── README.md
```


---

# Methodology

## Data Processing

Implemented using sklearn Pipeline:

- Numerical feature scaling
- Categorical feature one-hot encoding
- Unknown category handling


## Models

Compared:

1. Logistic Regression
2. Random Forest


---

# Data Leakage Analysis

The feature `duration` was excluded from the final model.

Reason:

`duration` represents the length of the last contact and is only available after the marketing call.

Using this feature would introduce information unavailable during real prediction.

Therefore, the final production model does not use `duration`.

---

# Model Comparison

Validation performance:

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.802 | 0.470 | 0.495 |
| Random Forest | 0.782 | 0.438 | 0.470 |


Logistic Regression achieved better performance and was selected as the final model.

---

# Threshold Optimization

Because the dataset is highly imbalanced, the default classification threshold (0.5) is not optimal.

The decision threshold was optimized using validation F1 score.

Best threshold: 0.2208


---

# Final Test Performance

Final model:

Logistic Regression


|Metric|Score|
|-|-:|
|ROC-AUC|0.805|
|PR-AUC|0.468|
|Precision|0.467|
|Recall|0.560|
|F1|0.509|

---

# Error Analysis

## False Positives

The model incorrectly predicted some customers as subscribers.

Common patterns:

- Similar demographic profiles to successful customers
- Limited historical campaign signals

This suggests that additional behavioral features could improve prediction.

---

## False Negatives

Some customers who subscribed were missed by the model.

Examples include:

- Retired customers
- Customers with limited campaign history

This indicates that customer intent cannot be fully captured by demographic and campaign information alone.

---

# Limitations and Future Work

Current features mainly include:

- Demographic information
- Campaign information
- Economic indicators


Future improvements:

- Account balance features
- Customer transaction history
- Previous product ownership
- Online banking behavior


---

# How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Train Model

```bash
python src/train.py
```

---

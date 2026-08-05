import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from models import build_logistic_model


from evaluate import (
    evaluate_model,
    find_best_threshold
)



# =========================
# Load data
# =========================

df = pd.read_csv(
    "data/bank-additional-full.csv",
    sep=";"
)



# =========================
# Target
# =========================

df["target"] = df["y"].map(
    {
        "no":0,
        "yes":1
    }
)



# =========================
# Remove leakage feature
# =========================

X = df.drop(
    columns=[
        "y",
        "target",
        "duration"
    ]
)


y = df["target"]



# =========================
# Split data
# =========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)



X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)



print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)



# =========================
# Build model
# =========================

model = build_logistic_model(
    X_train
)



# =========================
# Train
# =========================

model.fit(
    X_train,
    y_train
)



# =========================
# Validation threshold
# =========================

val_prob = model.predict_proba(
    X_val
)[:,1]


threshold_result = find_best_threshold(
    y_val,
    val_prob
)


best_threshold = threshold_result["threshold"]


print(
    "\nBest threshold:",
    best_threshold
)



# =========================
# Test evaluation
# =========================

test_prob = model.predict_proba(
    X_test
)[:,1]


test_result = evaluate_model(
    y_test,
    test_prob,
    threshold=best_threshold
)



print("\nTest Result:")

for key, value in test_result.items():

    print(
        f"{key}: {value:.4f}"
    )
joblib.dump(
    model,
    "models/logistic_model.pkl"
)
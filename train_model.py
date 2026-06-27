import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier

from preprocess import preprocess_data


# ==========================
# File Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ==========================
# Load Dataset
# ==========================

print(f"Loading dataset from:\n{DATA_PATH}")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found!\nExpected location:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)


# ==========================
# Preprocess Data
# ==========================

X, y = preprocess_data(df)


# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================
# Build Model
# ==========================

model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)


# ==========================
# Train Model
# ==========================

print("\nTraining Model...")

model.fit(X_train, y_train)


# ==========================
# Predictions
# ==========================

y_pred = model.predict(X_test)


# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Churn", "Churn"]
    )
)


# ==========================
# Cross Validation
# ==========================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print(
    f"\nCross Validation Accuracy: "
    f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
)


# ==========================
# Save Metrics
# ==========================

metrics = {
    "accuracy": round(accuracy, 4),
    "cv_mean": round(cv_scores.mean(), 4),
    "cv_std": round(cv_scores.std(), 4)
}

metrics_path = MODELS_DIR / "metrics.json"

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nMetrics saved to:\n{metrics_path}")


# ==========================
# Save Model
# ==========================

model_path = MODELS_DIR / "churn_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved to:\n{model_path}")

print("\nTraining Completed Successfully!")
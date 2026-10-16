
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

DATA_DIR = Path("research/data/UNSW-NB15")
RESULT_DIR = Path("research/results")


def main():
    print("Loading training dataset...")
    df = pd.read_csv(DATA_DIR / "UNSW_NB15_training-set.csv")

    y = df["label"].astype(int)
    X = df.drop(
        columns=["label", "id", "attack_cat"], errors="ignore"
    )

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )),
    ])

    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    print("Preprocessing training and validation data...")
    X_train = prep.fit_transform(X_train)
    X_val = prep.transform(X_val)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("ann", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=100,
            batch_size=512,
            early_stopping=False,
            random_state=42,
        )),
    ])

    print("Training ANN...")
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_val)[:, 1]

    rows = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_val, pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "Threshold": threshold,
            "Precision": precision_score(
                y_val, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_val, pred, zero_division=0
            ),
            "F1": f1_score(y_val, pred, zero_division=0),
            "False_Positive_Rate": fp / (fp + tn),
            "False_Positives": int(fp),
            "Missed_Attacks": int(fn),
            "True_Negatives": int(tn),
            "True_Positives": int(tp),
            "ROC_AUC": roc_auc_score(y_val, scores),
        })

    result = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "ann_threshold_analysis.csv"
    result.to_csv(out, index=False)

    print("\n=== ANN Validation Threshold Analysis ===")
    print(result.round(4).to_string(index=False))
    print(f"\nSaved to: {out}")
    print("Thresholds were evaluated on a validation split, not the test set.")


if __name__ == "__main__":
    main()

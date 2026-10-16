
from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

DATA_DIR = Path("research/data/UNSW-NB15")
RESULT_DIR = Path("research/results")


def main():
    train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

    print("Loading training and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target = "label"
    excluded = ["label", "id", "attack_cat"]

    y_train = train_df[target].astype(int)
    y_test = test_df[target].astype(int)

    X_train = train_df.drop(columns=excluded, errors="ignore")
    X_test = test_df.drop(columns=excluded, errors="ignore")

    # Keep the same feature columns and order
    X_test = X_test[X_train.columns]

    numeric_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = [
        col for col in X_train.columns
        if col not in numeric_cols
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    print("Fitting preprocessing on training data only...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

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

    print("Training ANN on training dataset...")
    start = time.perf_counter()
    model.fit(X_train_processed, y_train)
    training_seconds = time.perf_counter() - start

    print("Predicting on independent test dataset...")
    scores = model.predict_proba(X_test_processed)[:, 1]

    rows = []

    for threshold in [0.3, 0.4]:
        predictions = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "Model": "MLP_ANN",
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(y_test, scores),
            "PR_AUC": average_precision_score(y_test, scores),
            "False_Positive_Rate": fp / (fp + tn) if (fp + tn) else 0,
            "False_Positives": int(fp),
            "Missed_Attacks": int(fn),
            "True_Negatives": int(tn),
            "True_Positives": int(tp),
            "Training_Seconds": round(training_seconds, 3),
        })

    results = pd.DataFrame(rows)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULT_DIR / "ann_test_thresholds.csv"
    results.to_csv(output_path, index=False)

    print("\n=== ANN Independent Test-Set Threshold Evaluation ===")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved results to: {output_path}")
    print("Training time excludes preprocessing time.")
    print("Thresholds were chosen from validation analysis before this test.")
    print("Do not use these test results to repeatedly retune the thresholds.")


if __name__ == "__main__":
    main()


from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

DATA_DIR = Path("research/data/UNSW-NB15")
RESULT_DIR = Path("research/results")

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"


def main():
    print("Loading UNSW-NB15...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    # Exclude target, ID and attack category to avoid leakage.
    X_train = train_df.drop(
        columns=["label", "id", "attack_cat"], errors="ignore"
    )
    X_test = test_df.drop(
        columns=["label", "id", "attack_cat"], errors="ignore"
    ).reindex(columns=X_train.columns)

    numeric_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()
    categorical_cols = [
        col for col in X_train.columns if col not in numeric_cols
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    print("Preprocessing...")
    X_train = np.asarray(
        preprocessor.fit_transform(X_train), dtype=np.float32
    )
    X_test = np.asarray(
        preprocessor.transform(X_test), dtype=np.float32
    )

    # RBM input is scaled using training data only.
    scaler = MinMaxScaler()
    X_train = np.clip(scaler.fit_transform(X_train), 0, 1)
    X_test = np.clip(scaler.transform(X_test), 0, 1)

    # Learn normal traffic patterns only.
    normal_train = X_train[y_train.to_numpy() == 0]

    print("Training RBM on normal traffic...")
    rbm = BernoulliRBM(
        n_components=32,
        learning_rate=0.01,
        batch_size=256,
        n_iter=10,
        random_state=42,
        verbose=True,
    )

    start = time.perf_counter()
    rbm.fit(normal_train)

    normal_scores = -rbm.score_samples(normal_train)
    test_scores = -rbm.score_samples(X_test)
    training_seconds = time.perf_counter() - start

    # Thresholds are selected using NORMAL TRAINING data only.
    percentiles = [90, 95, 97, 99]
    rows = []

    for percentile in percentiles:
        threshold = np.percentile(normal_scores, percentile)
        predictions = (test_scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "Threshold_Percentile": percentile,
            "Threshold_Value": float(threshold),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "False_Positive_Rate": fp / (fp + tn) if (fp + tn) else 0,
            "False_Positives": int(fp),
            "Missed_Attacks": int(fn),
            "True_Negatives": int(tn),
            "True_Positives": int(tp),
            "ROC_AUC": roc_auc_score(y_test, test_scores),
            "RBM_Training_Seconds": training_seconds,
        })

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULT_DIR / "rbm_threshold_analysis.csv"

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_file, index=False)

    print("\n=== RBM Threshold Analysis ===")
    print(result_df.round(4).to_string(index=False))
    print(f"\nSaved results to: {output_file}")
    print(
        "\nThresholds are based only on normal training records. "
        "Testing labels are used only for evaluation."
    )


if __name__ == "__main__":
    main()

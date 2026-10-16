
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path("research/data/UNSW-NB15")
RESULT_DIR = Path("research/results")

TRAIN_FILE = DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = DATA_DIR / "UNSW_NB15_testing-set.csv"

TARGET = "label"
DROP_COLUMNS = ["id", "attack_cat"]


def calculate_metrics(name, y_true, y_pred, score, elapsed):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, score),
        "PR_AUC": average_precision_score(y_true, score),
        "False_Positive_Rate": fp / (fp + tn) if (fp + tn) else 0,
        "False_Positives": int(fp),
        "Missed_Attacks": int(fn),
        "True_Negatives": int(tn),
        "True_Positives": int(tp),
        "Runtime_Seconds": elapsed,
    }


def main():
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "UNSW-NB15 files were not found in research/data/UNSW-NB15."
        )

    print("Loading UNSW-NB15 datasets...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    for df in (train_df, test_df):
        df.columns = df.columns.str.strip()

    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise ValueError("Both datasets must contain the 'label' column.")

    if train_df[TARGET].isna().any() or test_df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError("Training labels must be 0 or 1.")
    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Testing labels must be 0 or 1.")

    # Exclude target and possible leakage/identifier columns.
    X_train = train_df.drop(
        columns=[TARGET] + DROP_COLUMNS, errors="ignore"
    )
    X_test = test_df.drop(
        columns=[TARGET] + DROP_COLUMNS, errors="ignore"
    )

    # Use matching feature columns in the same order.
    X_test = X_test.reindex(columns=X_train.columns)

    numeric_cols = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()
    categorical_cols = [
        c for c in X_train.columns if c not in numeric_cols
    ]

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print("Training label distribution:")
    print(y_train.value_counts().sort_index().to_dict())
    print("Testing label distribution:")
    print(y_test.value_counts().sort_index().to_dict())

    # Fit all imputers, encoders and scalers on training data only.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    print("\nFitting preprocessing on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.asarray(
        X_train_processed, dtype=np.float32
    )
    X_test_processed = np.asarray(
        X_test_processed, dtype=np.float32
    )

    if not np.isfinite(X_train_processed).all():
        raise ValueError("Non-finite values found in processed training data.")
    if not np.isfinite(X_test_processed).all():
        raise ValueError("Non-finite values found in processed test data.")

    print("Processed feature count:", X_train_processed.shape[1])

    results = []

    # 1. Supervised ANN baseline.
    print("\n[1/3] Training ANN...")
    start = time.perf_counter()

    ann = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=100,
            early_stopping=False,
            random_state=42,
            batch_size=512,
            learning_rate_init=0.001,
        )),
    ])

    ann.fit(X_train_processed, y_train)
    ann_score = ann.predict_proba(X_test_processed)[:, 1]
    ann_pred = (ann_score >= 0.5).astype(int)

    results.append(calculate_metrics(
        "MLP_ANN", y_test, ann_pred, ann_score,
        time.perf_counter() - start
    ))

    # RBM requires inputs in [0, 1].
    # Fit the RBM scaler on training rows only.
    print("\nPreparing data for RBM...")
    rbm_scaler = MinMaxScaler()
    X_train_rbm = np.clip(
        rbm_scaler.fit_transform(X_train_processed), 0, 1
    )
    X_test_rbm = np.clip(
        rbm_scaler.transform(X_test_processed), 0, 1
    )

    # 2. Unsupervised anomaly detection.
    # Learn normal traffic patterns using normal training rows only.
    print("\n[2/3] Training RBM anomaly detector...")
    normal_mask = (y_train.to_numpy() == 0)
    X_normal = X_train_rbm[normal_mask]

    rbm = BernoulliRBM(
        n_components=32,
        learning_rate=0.01,
        batch_size=256,
        n_iter=10,
        random_state=42,
        verbose=True,
    )

    start = time.perf_counter()
    rbm.fit(X_normal)

    normal_anomaly_scores = -rbm.score_samples(X_normal)
    test_anomaly_scores = -rbm.score_samples(X_test_rbm)

    # Set threshold from normal training data only.
    threshold = np.quantile(normal_anomaly_scores, 0.95)
    rbm_pred = (test_anomaly_scores >= threshold).astype(int)

    # Larger anomaly score indicates a stronger attack suspicion.
    results.append(calculate_metrics(
        "RBM_Anomaly", y_test, rbm_pred, test_anomaly_scores,
        time.perf_counter() - start
    ))

    # 3. RBM hidden features + supervised logistic regression.
    print("\n[3/3] Training RBM + Logistic Regression hybrid...")
    start = time.perf_counter()

    hybrid_train_features = rbm.transform(X_train_rbm)
    hybrid_test_features = rbm.transform(X_test_rbm)

    hybrid = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    )

    hybrid.fit(hybrid_train_features, y_train)
    hybrid_score = hybrid.predict_proba(hybrid_test_features)[:, 1]
    hybrid_pred = (hybrid_score >= 0.5).astype(int)

    results.append(calculate_metrics(
        "RBM_ANN_Hybrid", y_test, hybrid_pred, hybrid_score,
        time.perf_counter() - start
    ))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULT_DIR / "ann_rbm_comparison.csv", index=False
    )

    # Save confusion matrices in a separate, easy-to-read CSV.
    matrix_rows = []
    for row in results:
        name = row["Model"]
        tn = row["True_Negatives"]
        fp = row["False_Positives"]
        fn = row["Missed_Attacks"]
        tp = row["True_Positives"]

        matrix_rows.extend([
            {"Model": name, "Actual": "Normal", "Predicted": "Normal",
             "Count": tn},
            {"Model": name, "Actual": "Normal", "Predicted": "Attack",
             "Count": fp},
            {"Model": name, "Actual": "Attack", "Predicted": "Normal",
             "Count": fn},
            {"Model": name, "Actual": "Attack", "Predicted": "Attack",
             "Count": tp},
        ])

    pd.DataFrame(matrix_rows).to_csv(
        RESULT_DIR / "confusion_matrices.csv", index=False
    )

    print("\n=== ANN vs RBM vs Hybrid ===")
    print(results_df.round(4).to_string(index=False))
    print("\nSaved:")
    print(RESULT_DIR / "ann_rbm_comparison.csv")
    print(RESULT_DIR / "confusion_matrices.csv")
    print(
        "\nNote: These are experimental results on UNSW-NB15. "
        "They do not establish real-world security performance."
    )


if __name__ == "__main__":
    main()

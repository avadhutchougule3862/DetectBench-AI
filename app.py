import csv
import io
from flask import make_response
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

FEATURES = [
    "failed_logins",
    "requests_per_minute",
    "unusual_hour",
    "new_device",
    "bytes_sent_kb"
]


def evaluate_data(data):
    data = data.copy()

    # Rule-Based Detection
    rule_alert = (
        (data["failed_logins"] >= 5)
        | (data["requests_per_minute"] >= 45)
        | (
            (data["unusual_hour"] == 1)
            & (data["new_device"] == 1)
        )
    ).astype(int)

    # Isolation Forest Detection
    model = IsolationForest(
        n_estimators=100,
        contamination=0.10,
        random_state=42
    )

    ml_alert = (
        model.fit_predict(data[FEATURES]) == -1
    ).astype(int)

    # Hybrid Detection
    hybrid_alert = (
        (rule_alert == 1) | (ml_alert == 1)
    ).astype(int)

    # Add predictions to the dataset
    data["rule_alert"] = rule_alert
    data["ml_alert"] = ml_alert
    data["hybrid_alert"] = hybrid_alert

    predictions = {
        "Rule-Based": rule_alert,
        "Isolation Forest": ml_alert,
        "Hybrid": hybrid_alert
    }

    results = {}

    # Calculate metrics only when actual labels are available
    if "actual_label" in data.columns:
        actual = data["actual_label"].astype(int)

        if not actual.isin([0, 1]).all():
            raise ValueError(
                "actual_label must contain only 0 and 1."
            )

        for name, predicted in predictions.items():

            # Confusion Matrix: labels are [0, 1]
            tn, fp, fn, tp = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            ).ravel()

            # Precision, Recall and F1-score
            precision = precision_score(
                actual, predicted, zero_division=0
            )

            recall = recall_score(
                actual, predicted, zero_division=0
            )

            f1 = f1_score(
                actual, predicted, zero_division=0
            )

            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            results[name] = {
                "alerts": int(predicted.sum()),
                "precision": round(precision * 100, 1),
                "recall": round(recall * 100, 1),
                "f1": round(f1 * 100, 1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "fpr": round(fpr * 100, 1),
                "false_positives": int(fp),
                "missed": int(fn)
            }

    else:
        # Without ground-truth labels, metrics cannot be calculated
        for name, predicted in predictions.items():
            results[name] = {
                "alerts": int(predicted.sum())
            }

    return data, results

def run_demo():
    rng = np.random.default_rng(42)
    n = 300

    data = pd.DataFrame({
        "failed_logins": rng.poisson(1.5, n),
        "requests_per_minute": rng.poisson(20, n),
        "unusual_hour": rng.integers(0, 2, n),
        "new_device": rng.integers(0, 2, n),
        "bytes_sent_kb": rng.exponential(500, n)
    })

    risk_score = (
        (data["failed_logins"] >= 6).astype(int)
        + (data["requests_per_minute"] >= 42).astype(int)
        + (
            (data["unusual_hour"] == 1)
            & (data["new_device"] == 1)
        ).astype(int)
        + (data["bytes_sent_kb"] >= 1800).astype(int)
    )

    data["actual_label"] = (risk_score >= 2).astype(int)

    events, results = evaluate_data(data)
    return events, results


@app.route("/", methods=["GET", "POST"])
def dashboard():
    demo_events, demo_results = run_demo()

    upload_results = None
    upload_events = None
    upload_error = None
    upload_has_labels = False

    if request.method == "POST":
        file = request.files.get("csv_file")

        if not file or not file.filename:
            upload_error = "Please select a CSV file."

        elif not file.filename.lower().endswith(".csv"):
            upload_error = "Only CSV files are supported."

        else:
            try:
                uploaded = pd.read_csv(file)

                if uploaded.empty:
                    raise ValueError("The uploaded CSV is empty.")

                if len(uploaded) > 50000:
                    raise ValueError(
                        "Please upload a CSV with no more than 50,000 rows."
                    )

                missing = [col for col in FEATURES if col not in uploaded.columns]

                if missing:
                    raise ValueError(
                        "Missing required columns: " + ", ".join(missing)
                    )

                for col in FEATURES:
                    uploaded[col] = pd.to_numeric(
                        uploaded[col], errors="raise"
                    )

                if uploaded[FEATURES].isnull().any().any():
                    raise ValueError(
                        "Required feature columns contain missing values."
                    )

                if not np.isfinite(uploaded[FEATURES].to_numpy()).all():
                    raise ValueError(
                        "Feature columns contain infinite or invalid values."
                    )

                for col in ["unusual_hour", "new_device"]:
                    if not uploaded[col].isin([0, 1]).all():
                        raise ValueError(
                            f"{col} must contain only 0 or 1."
                        )

                if (uploaded[["failed_logins", "requests_per_minute",
                              "bytes_sent_kb"]] < 0).any().any():
                    raise ValueError(
                        "Counts and bytes_sent_kb cannot be negative."
                    )

                if "actual_label" in uploaded.columns:
                    uploaded["actual_label"] = pd.to_numeric(
                        uploaded["actual_label"], errors="raise"
                    )

                    if uploaded["actual_label"].isnull().any() or not uploaded[
                        "actual_label"
                    ].isin([0, 1]).all():
                        raise ValueError(
                            "actual_label must contain only 0 and 1."
                        )

                    upload_has_labels = True

                upload_events, upload_results = evaluate_data(uploaded)

                # Show only a limited number of rows on the page
                upload_events = upload_events.tail(20).to_dict(
                    orient="records"
                )

            except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
                upload_error = f"Could not process CSV: {exc}"
            except Exception:
                app.logger.exception("CSV processing failed")
                upload_error = (
                    "The CSV could not be processed. Check its format and values."
                )

    chart_data = {
        "labels": list(demo_results.keys()),
        "precision": [
            v.get("precision", 0) for v in demo_results.values()
        ],
        "recall": [
            v.get("recall", 0) for v in demo_results.values()
        ],
        "f1": [
            v.get("f1", 0) for v in demo_results.values()
        ],
        "fpr": [
            v.get("fpr", 0) for v in demo_results.values()
        ]
    }

    return render_template(
        "index.html",
        results=demo_results,
        total=len(demo_events),
        suspicious=int(demo_events["actual_label"].sum()),
        events=demo_events.tail(12).to_dict(orient="records"),
        chart_data=chart_data,
        upload_results=upload_results,
        upload_events=upload_events,
        upload_error=upload_error,
        upload_has_labels=upload_has_labels
    )


@app.errorhandler(413)
def file_too_large(error):
    return "CSV file is too large. Maximum size is 5 MB.", 413


@app.route("/download-report", methods=["POST"])
def download_report():
    import io
    import csv
    from flask import make_response

    if "csv_file" not in request.files:
        return "Please upload a CSV file.", 400

    file = request.files["csv_file"]

    if not file.filename or not file.filename.lower().endswith(".csv"):
        return "Please upload a valid CSV file.", 400

    try:
        data = pd.read_csv(file)

        if data.empty:
            return "The uploaded CSV file is empty.", 400

        required_columns = FEATURES
        missing_columns = [
            column for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:
            return (
                "Missing required columns: "
                + ", ".join(missing_columns)
            ), 400

        # Use the same data analysis function as the dashboard.
        analyzed_data, results = evaluate_data(data)

        # Build one CSV report with event details and model metrics.
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["DETECTBENCH AI - ANALYSIS REPORT"])
        writer.writerow([])
        writer.writerow(["MODEL METRICS"])

        if results and all(
            metric in next(iter(results.values()))
            for metric in ["precision", "recall", "f1"]
        ):
            writer.writerow([
                "Model", "Alerts", "Precision (%)", "Recall (%)",
                "F1 Score (%)", "TN", "FP", "FN", "TP"
            ])

            for model_name, metrics in results.items():
                writer.writerow([
                    model_name,
                    metrics.get("alerts", 0),
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1", 0),
                    metrics.get("tn", 0),
                    metrics.get("fp", 0),
                    metrics.get("fn", 0),
                    metrics.get("tp", 0)
                ])
        else:
            writer.writerow(["Model", "Alerts"])
            for model_name, metrics in results.items():
                writer.writerow([
                    model_name,
                    metrics.get("alerts", 0)
                ])

        writer.writerow([])
        writer.writerow(["EVENT ANALYSIS"])

        writer.writerow(list(analyzed_data.columns))
        for _, row in analyzed_data.iterrows():
            writer.writerow(row.tolist())

        response = make_response(output.getvalue())
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers[
            "Content-Disposition"
        ] = "attachment; filename=detectbench_analysis_report.csv"

        return response

    except (ValueError, KeyError, pd.errors.ParserError) as error:
        return f"Could not analyze CSV: {error}", 400


if __name__ == "__main__":
    app.run(debug=False)
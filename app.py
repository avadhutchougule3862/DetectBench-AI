import csv
import io
from flask import make_response
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

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

        for name, predicted in predictions.items():
            fp = int(((predicted == 1) & (actual == 0)).sum())
            tn = int(((predicted == 0) & (actual == 0)).sum())
            missed = int(((predicted == 0) & (actual == 1)).sum())

            results[name] = {
                "alerts": int(predicted.sum()),
                "precision": round(
                    precision_score(actual, predicted, zero_division=0) * 100, 1
                ),
                "recall": round(
                    recall_score(actual, predicted, zero_division=0) * 100, 1
                ),
                "f1": round(
                    f1_score(actual, predicted, zero_division=0) * 100, 1
                ),
                "fpr": round(fp / max(fp + tn, 1) * 100, 1),
                "false_positives": fp,
                "missed": missed
            }

    else:
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
@app.route("/download-report", methods=["POST"])
def download_report():
    file = request.files.get("csv_file")

    if not file or not file.filename:
        return "Please select the CSV file again to download the report.", 400

    if not file.filename.lower().endswith(".csv"):
        return "Only CSV files are supported.", 400

    try:
        uploaded = pd.read_csv(file)

        if uploaded.empty or len(uploaded) > 50000:
            return "CSV must contain 1 to 50,000 rows.", 400

        missing = [col for col in FEATURES if col not in uploaded.columns]
        if missing:
            return "Missing required columns: " + ", ".join(missing), 400

        for col in FEATURES:
            uploaded[col] = pd.to_numeric(uploaded[col], errors="raise")

        if uploaded[FEATURES].isnull().any().any():
            return "Required columns contain missing values.", 400

        if not np.isfinite(uploaded[FEATURES].to_numpy()).all():
            return "Feature columns contain invalid values.", 400

        for col in ["unusual_hour", "new_device"]:
            if not uploaded[col].isin([0, 1]).all():
                return f"{col} must contain only 0 or 1.", 400

        if (uploaded[["failed_logins", "requests_per_minute",
                      "bytes_sent_kb"]] < 0).any().any():
            return "Counts and bytes_sent_kb cannot be negative.", 400

        if "actual_label" in uploaded.columns:
            uploaded["actual_label"] = pd.to_numeric(
                uploaded["actual_label"], errors="raise"
            )
            if (
                uploaded["actual_label"].isnull().any()
                or not uploaded["actual_label"].isin([0, 1]).all()
            ):
                return "actual_label must contain only 0 and 1.", 400

        analyzed, results = evaluate_data(uploaded)

        output = io.StringIO()
        output.write("DETECTBENCH AI - ANALYSIS REPORT\n")
        output.write("Note: Results depend on the uploaded dataset.\n\n")
        output.write("EVENT ANALYSIS\n")
        analyzed.to_csv(output, index=False)

        output.write("\nDETECTION METRICS\n")
        metrics_df = pd.DataFrame.from_dict(results, orient="index")
        metrics_df.index.name = "detection_method"
        metrics_df.to_csv(output)

        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = (
            "attachment; filename=detectbench_analysis_report.csv"
        )
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        return response

    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return f"Could not process CSV: {exc}", 400

@app.errorhandler(413)
def file_too_large(error):
    return "CSV file is too large. Maximum size is 5 MB.", 413


if __name__ == "__main__":
    app.run(debug=True)
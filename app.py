
import csv
import io
import os

import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    make_response,
)
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from werkzeug.utils import secure_filename

from extensions import db, migrate
from models import User, AnalysisHistory


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "dev-only-change-this-secret-key",
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

database_url = os.environ.get(
    "DATABASE_URL",
    "sqlite:///detectbench.db",
)

if database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://",
        "postgresql+psycopg://",
        1,
    )
elif database_url.startswith("postgresql://"):
    database_url = database_url.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1,
    )

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
migrate.init_app(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------
# ML FEATURES
# --------------------------------------------------

FEATURES = [
    "failed_logins",
    "requests_per_minute",
    "unusual_hour",
    "new_device",
    "bytes_sent_kb",
]


# --------------------------------------------------
# DETECTION FUNCTION
# --------------------------------------------------

def evaluate_data(data):
    data = data.copy()

    rule_alert = (
        (data["failed_logins"] >= 5)
        | (data["requests_per_minute"] >= 45)
        | (
            (data["unusual_hour"] == 1)
            & (data["new_device"] == 1)
        )
    ).astype(int)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.10,
        random_state=42,
    )

    ml_alert = (
        model.fit_predict(data[FEATURES]) == -1
    ).astype(int)

    hybrid_alert = (
        (rule_alert == 1) | (ml_alert == 1)
    ).astype(int)

    data["rule_alert"] = rule_alert
    data["ml_alert"] = ml_alert
    data["hybrid_alert"] = hybrid_alert

    predictions = {
        "Rule-Based": rule_alert,
        "Isolation Forest": ml_alert,
        "Hybrid": hybrid_alert,
    }

    results = {}

    if "actual_label" in data.columns:
        actual = data["actual_label"].astype(int)

        if not actual.isin([0, 1]).all():
            raise ValueError(
                "actual_label must contain only 0 and 1."
            )

        for name, predicted in predictions.items():
            tn, fp, fn, tp = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1],
            ).ravel()

            precision = precision_score(
                actual, predicted, zero_division=0
            )
            recall = recall_score(
                actual, predicted, zero_division=0
            )
            f1 = f1_score(
                actual, predicted, zero_division=0
            )

            fpr = (
                fp / (fp + tn)
                if (fp + tn) > 0
                else 0.0
            )

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
                "missed": int(fn),
            }

    else:
        for name, predicted in predictions.items():
            results[name] = {
                "alerts": int(predicted.sum()),
            }

    return data, results


# --------------------------------------------------
# DEMO DATA
# --------------------------------------------------

def run_demo():
    rng = np.random.default_rng(42)
    n = 300

    data = pd.DataFrame({
        "failed_logins": rng.poisson(1.5, n),
        "requests_per_minute": rng.poisson(20, n),
        "unusual_hour": rng.integers(0, 2, n),
        "new_device": rng.integers(0, 2, n),
        "bytes_sent_kb": rng.exponential(500, n),
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


# --------------------------------------------------
# REGISTER
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if len(username) < 3 or len(username) > 80:
            flash(
                "Username must be between 3 and 80 characters.",
                "error",
            )
            return render_template("register.html")

        if not email or len(email) > 120 or "@" not in email:
            flash("Please enter a valid email address.", "error")
            return render_template("register.html")

        if len(password) < 8:
            flash(
                "Password must be at least 8 characters.",
                "error",
            )
            return render_template("register.html")

        existing_user = db.session.execute(
            db.select(User).where(
                db.or_(
                    User.username == username,
                    User.email == email,
                )
            )
        ).scalar_one_or_none()

        if existing_user:
            flash(
                "Username or email already exists.",
                "error",
            )
            return render_template("register.html")

        user = User(username=username, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash(
            "Registration successful! Please log in.",
            "success",
        )
        return redirect(url_for("login"))

    return render_template("register.html")


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = db.session.execute(
            db.select(User).where(User.email == email)
        ).scalar_one_or_none()

        if user is not None and user.check_password(password):
            login_user(user)

            # Explicitly mark the session as modified so Flask
            # persists the authenticated session cookie.
            from flask import session
            session.modified = True

            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully.", "success")
    return redirect(url_for("login"))


# --------------------------------------------------
# DASHBOARD AND CSV UPLOAD
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
@login_required
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

                missing = [
                    col for col in FEATURES
                    if col not in uploaded.columns
                ]

                if missing:
                    raise ValueError(
                        "Missing required columns: "
                        + ", ".join(missing)
                    )

                for col in FEATURES:
                    uploaded[col] = pd.to_numeric(
                        uploaded[col],
                        errors="raise",
                    )

                if uploaded[FEATURES].isnull().any().any():
                    raise ValueError(
                        "Required feature columns contain missing values."
                    )

                if not np.isfinite(
                    uploaded[FEATURES].to_numpy()
                ).all():
                    raise ValueError(
                        "Feature columns contain infinite or invalid values."
                    )

                for col in ["unusual_hour", "new_device"]:
                    if not uploaded[col].isin([0, 1]).all():
                        raise ValueError(
                            f"{col} must contain only 0 or 1."
                        )

                nonnegative_cols = [
                    "failed_logins",
                    "requests_per_minute",
                    "bytes_sent_kb",
                ]

                if (
                    uploaded[nonnegative_cols] < 0
                ).any().any():
                    raise ValueError(
                        "Counts and bytes_sent_kb cannot be negative."
                    )

                if "actual_label" in uploaded.columns:
                    uploaded["actual_label"] = pd.to_numeric(
                        uploaded["actual_label"],
                        errors="raise",
                    )

                    if (
                        uploaded["actual_label"].isnull().any()
                        or not uploaded["actual_label"].isin([0, 1]).all()
                    ):
                        raise ValueError(
                            "actual_label must contain only 0 and 1."
                        )

                    upload_has_labels = True

                upload_events, upload_results = evaluate_data(uploaded)

                history_record = AnalysisHistory(
                    user_id=current_user.id,
                    filename=(
                        secure_filename(file.filename)
                        or "uploaded.csv"
                    ),
                    total_rows=len(uploaded),
                    rule_alerts=upload_results["Rule-Based"]["alerts"],
                    ml_alerts=upload_results["Isolation Forest"]["alerts"],
                    hybrid_alerts=upload_results["Hybrid"]["alerts"],
                )

                db.session.add(history_record)
                db.session.commit()

                upload_events = upload_events.tail(20).to_dict(
                    orient="records"
                )

            except (
                ValueError,
                pd.errors.ParserError,
                UnicodeDecodeError,
            ) as exc:
                db.session.rollback()
                upload_error = f"Could not process CSV: {exc}"

            except Exception:
                db.session.rollback()
                app.logger.exception("CSV processing failed")
                upload_error = (
                    "The CSV could not be processed. "
                    "Check its format and values."
                )

    chart_data = {
        "labels": list(demo_results.keys()),
        "precision": [
            value.get("precision", 0)
            for value in demo_results.values()
        ],
        "recall": [
            value.get("recall", 0)
            for value in demo_results.values()
        ],
        "f1": [
            value.get("f1", 0)
            for value in demo_results.values()
        ],
        "fpr": [
            value.get("fpr", 0)
            for value in demo_results.values()
        ],
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
        upload_has_labels=upload_has_labels,
    )


# --------------------------------------------------
# FILE SIZE ERROR
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return "CSV file is too large. Maximum size is 5 MB.", 413


# --------------------------------------------------
# DOWNLOAD ANALYSIS REPORT
# --------------------------------------------------

@app.route("/download-report", methods=["POST"])
@login_required
def download_report():
    file = request.files.get("csv_file")

    if file is None:
        return "Please upload a CSV file.", 400

    if not file.filename or not file.filename.lower().endswith(".csv"):
        return "Please upload a valid CSV file.", 400

    try:
        data = pd.read_csv(file)

        if data.empty:
            return "The uploaded CSV file is empty.", 400

        missing = [
            column for column in FEATURES
            if column not in data.columns
        ]

        if missing:
            return (
                "Missing required columns: " + ", ".join(missing)
            ), 400

        analyzed_data, results = evaluate_data(data)

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["DETECTBENCH AI - ANALYSIS REPORT"])
        writer.writerow([])
        writer.writerow(["MODEL METRICS"])

        has_metrics = bool(results) and all(
            all(
                key in metrics
                for key in ["precision", "recall", "f1"]
            )
            for metrics in results.values()
        )

        if has_metrics:
            writer.writerow([
                "Model", "Alerts", "Precision (%)", "Recall (%)",
                "F1 Score (%)", "TN", "FP", "FN", "TP",
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
                    metrics.get("tp", 0),
                ])
        else:
            writer.writerow(["Model", "Alerts"])

            for model_name, metrics in results.items():
                writer.writerow([
                    model_name,
                    metrics.get("alerts", 0),
                ])

        writer.writerow([])
        writer.writerow(["EVENT ANALYSIS"])
        writer.writerow(list(analyzed_data.columns))

        for _, row in analyzed_data.iterrows():
            writer.writerow(row.tolist())

        response = make_response(output.getvalue())
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["Content-Disposition"] = (
            "attachment; filename=detectbench_analysis_report.csv"
        )

        return response

    except (
        ValueError,
        KeyError,
        pd.errors.ParserError,
    ) as error:
        return f"Could not analyze CSV: {error}", 400


# --------------------------------------------------
# USER-SPECIFIC ANALYSIS HISTORY
# --------------------------------------------------

@app.route("/history")
@login_required
def analysis_history():
    history = db.session.execute(
        db.select(AnalysisHistory)
        .where(AnalysisHistory.user_id == current_user.id)
        .order_by(AnalysisHistory.created_at.desc())
        .limit(100)
    ).scalars().all()

    return render_template("history.html", history=history)


# --------------------------------------------------
# LOCAL DEVELOPMENT
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
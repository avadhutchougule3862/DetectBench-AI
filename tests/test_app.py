
import io
import uuid

import pandas as pd
import pytest

from app import app, evaluate_data, load_research_results
app.config["WTF_CSRF_ENABLED"] = False
from extensions import db
from models import User, AnalysisHistory


TEST_DATABASE_URI = "sqlite:///detectbench_test.db"


@pytest.fixture(autouse=True)
def setup_test_database():
    database_uri = app.config["SQLALCHEMY_DATABASE_URI"]

    # Never allow tests to modify a production database.
    if database_uri != TEST_DATABASE_URI:
        pytest.fail(
            "Tests must use DATABASE_URL="
            "'sqlite:///detectbench_test.db'"
        )

    with app.app_context():
        db.create_all()

    yield

    with app.app_context():
        db.session.remove()
        db.drop_all()


def create_test_user(password="testpass123"):
    unique = uuid.uuid4().hex[:10]

    with app.app_context():
        user = User(
            username=f"user_{unique}",
            email=f"{unique}@example.com",
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        return user.id


def get_user_email(user_id):
    with app.app_context():
        user = db.session.get(User, user_id)
        assert user is not None
        return user.email


def login_user(client, email, password="testpass123"):
    # Do not follow redirects; inspect the original login response.
    return client.post(
        "/login",
        data={"email": email, "password": password},
        follow_redirects=False,
    )


def test_evaluate_data_returns_detection_columns():
    data = pd.DataFrame({
        "failed_logins": [0, 1, 8, 2, 0, 1],
        "requests_per_minute": [10, 15, 60, 20, 12, 18],
        "unusual_hour": [0, 0, 1, 0, 0, 1],
        "new_device": [0, 0, 1, 0, 0, 0],
        "bytes_sent_kb": [100, 200, 500, 300, 150, 250],
    })

    analyzed, results = evaluate_data(data)

    assert len(analyzed) == len(data)

    assert {
        "rule_alert",
        "ml_alert",
        "hybrid_alert",
    }.issubset(analyzed.columns)

    assert {
        "Rule-Based",
        "Isolation Forest",
        "Hybrid",
    }.issubset(results.keys())


def test_rule_detects_multiple_failed_logins():
    data = pd.DataFrame({
        "failed_logins": [6, 0, 1, 0, 0, 1],
        "requests_per_minute": [10, 10, 10, 10, 10, 10],
        "unusual_hour": [0, 0, 0, 0, 0, 0],
        "new_device": [0, 0, 0, 0, 0, 0],
        "bytes_sent_kb": [100, 100, 100, 100, 100, 100],
    })

    analyzed, _ = evaluate_data(data)

    assert analyzed.loc[0, "rule_alert"] == 1


def test_dashboard_requires_login():
    response = app.test_client().get("/")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_upload_rejects_missing_columns():
    user_id = create_test_user()
    client = app.test_client()

    login_response = login_user(
        client,
        get_user_email(user_id),
    )

    assert login_response.status_code == 302
    assert login_response.headers.get("Location") == "/"

    response = client.post(
        "/",
        data={
            "csv_file": (
                io.BytesIO(b"wrong_column\n1\n2\n"),
                "test.csv",
            )
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Missing required columns" in response.data


def test_download_report_returns_csv():
    user_id = create_test_user()
    client = app.test_client()

    login_response = login_user(
        client,
        get_user_email(user_id),
    )

    assert login_response.status_code == 302

    csv_content = (
        b"failed_logins,requests_per_minute,"
        b"unusual_hour,new_device,bytes_sent_kb\n"
        b"1,10,0,0,100\n"
        b"2,20,1,1,200\n"
        b"0,15,0,0,150\n"
        b"1,12,0,1,120\n"
        b"0,18,1,0,180\n"
        b"2,25,0,0,250\n"
    )

    response = client.post(
        "/download-report",
        data={
            "csv_file": (
                io.BytesIO(csv_content),
                "test.csv",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert b"rule_alert" in response.data
    assert b"hybrid_alert" in response.data


def test_history_requires_login():
    response = app.test_client().get("/history")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_download_report_requires_login():
    response = app.test_client().post("/download-report")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_history_isolated_between_two_users():
    user_a_id = create_test_user()
    user_b_id = create_test_user()

    email_a = get_user_email(user_a_id)
    email_b = get_user_email(user_b_id)

    with app.app_context():
        db.session.add_all([
            AnalysisHistory(
                user_id=user_a_id,
                filename="user_a.csv",
                total_rows=10,
                rule_alerts=2,
                ml_alerts=1,
                hybrid_alerts=2,
            ),
            AnalysisHistory(
                user_id=user_b_id,
                filename="user_b.csv",
                total_rows=20,
                rule_alerts=4,
                ml_alerts=3,
                hybrid_alerts=4,
            ),
        ])

        db.session.commit()

    client_a = app.test_client()
    client_b = app.test_client()

    response_login_a = login_user(client_a, email_a)
    response_login_b = login_user(client_b, email_b)

    # Both users should successfully log in.
    assert response_login_a.status_code == 302, (
        f"User A login failed: {response_login_a.status_code}"
    )
    assert response_login_b.status_code == 302, (
        f"User B login failed: {response_login_b.status_code}"
    )

    assert response_login_a.headers.get("Location") == "/"
    assert response_login_b.headers.get("Location") == "/"

    # Confirm that each client has its own authenticated session.
    with client_a.session_transaction() as session_a:
        session_user_a = session_a.get("_user_id")

    with client_b.session_transaction() as session_b:
        session_user_b = session_b.get("_user_id")

    assert session_user_a == str(user_a_id), (
        f"User A session mismatch: {session_user_a}"
    )
    assert session_user_b == str(user_b_id), (
        f"User B session mismatch: {session_user_b}"
    )

    # Fetch each user's history independently.
    response_a = client_a.get("/history")
    response_b = client_b.get("/history")

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    # User A must not see User B's history.
    assert b"user_a.csv" in response_a.data
    assert b"user_b.csv" not in response_a.data

    # User B must not see User A's history.
    assert b"user_b.csv" in response_b.data
    assert b"user_a.csv" not in response_b.data


def test_offline_research_results_are_loaded_from_csv():
    models, thresholds = load_research_results()

    assert len(models) == 3
    assert {row["model"] for row in models} == {
        "MLP_ANN", "RBM_Anomaly", "RBM_Logistic_Hybrid"
    }
    assert len(thresholds) == 2
    assert {row["threshold"] for row in thresholds} == {0.3, 0.4}
    ann = next(row for row in models if row["model"] == "MLP_ANN")
    assert ann["roc_auc"] == pytest.approx(97.5791, abs=0.01)

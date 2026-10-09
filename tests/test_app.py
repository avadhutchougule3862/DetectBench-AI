
import pandas as pd
from app import app, evaluate_data, FEATURES


def test_evaluate_data_returns_detection_columns():
    data = pd.DataFrame({
        "failed_logins": [0, 1, 8, 2, 0, 1],
        "requests_per_minute": [10, 15, 60, 20, 12, 18],
        "unusual_hour": [0, 0, 1, 0, 0, 1],
        "new_device": [0, 0, 1, 0, 0, 0],
        "bytes_sent_kb": [100, 200, 500, 300, 150, 250]
    })

    analyzed, results = evaluate_data(data)

    assert len(analyzed) == len(data)
    assert "rule_alert" in analyzed.columns
    assert "ml_alert" in analyzed.columns
    assert "hybrid_alert" in analyzed.columns
    assert "Rule-Based" in results
    assert "Isolation Forest" in results
    assert "Hybrid" in results


def test_rule_detects_multiple_failed_logins():
    data = pd.DataFrame({
        "failed_logins": [6, 0, 1, 0, 0, 1],
        "requests_per_minute": [10, 10, 10, 10, 10, 10],
        "unusual_hour": [0, 0, 0, 0, 0, 0],
        "new_device": [0, 0, 0, 0, 0, 0],
        "bytes_sent_kb": [100, 100, 100, 100, 100, 100]
    })

    analyzed, _ = evaluate_data(data)

    assert analyzed.loc[0, "rule_alert"] == 1


def test_dashboard_loads():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    def test_upload_rejects_missing_columns():
        client = app.test_client()

    response = client.post(
        "/",
        data={
            "csv_file": (
                __import__("io").BytesIO(
                    b"wrong_column\n1\n2\n"
                ),
                "test.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Missing required columns" in response.data


def test_download_report_returns_csv():
    import io

    client = app.test_client()

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
                "test.csv"
            )
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert b"rule_alert" in response.data
    assert b"hybrid_alert" in response.data
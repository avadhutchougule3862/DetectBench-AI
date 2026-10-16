# DetectBench AI

DetectBench AI is a Flask-based security event analysis dashboard that compares rule-based detection, Isolation Forest anomaly detection, and a hybrid detection approach.

## Features

- User registration, login, and logout
- Password hashing and authenticated dashboard access
- CSV upload and validation
- Rule-Based, Isolation Forest, and Hybrid alert detection
- Analysis history stored per user
- CSV analysis report download
- Precision, Recall, F1-score, and confusion-matrix metrics when actual labels are provided
- Offline research results dashboard using precomputed model comparison and ANN threshold results
- SQLite for local development and PostgreSQL for production
- Database migrations with Flask-Migrate
- Automated tests with pytest

## Technology Stack

- Python 3.12
- Flask
- Flask-SQLAlchemy and Flask-Migrate
- Flask-Login and Flask-WTF
- pandas, NumPy, and scikit-learn
- SQLite / PostgreSQL
- pytest
- Gunicorn for deployment

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/avadhutchougoule3862/DetectBench-AI.git
cd DetectBench-AI
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure local environment

For local development, set a secret key in PowerShell:

```powershell
$env:SECRET_KEY = "replace-with-a-long-random-development-secret"
```

The application uses its configured local SQLite database by default when `DATABASE_URL` is not set.

### 5. Apply database migrations

```powershell
flask db upgrade
```

### 6. Start the application

```powershell
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## CSV Input Format

The uploaded CSV must contain these columns:

- `failed_logins`
- `requests_per_minute`
- `unusual_hour`
- `new_device`
- `bytes_sent_kb`

Optional column:

- `actual_label` — binary ground-truth labels (`0` or `1`) used to calculate evaluation metrics.

The application validates the uploaded data and limits uploads to 5 MB and 50,000 rows.

A sample input file is available at `sample_events.csv`.

## Detection Methods

- **Rule-Based:** flags events based on predefined security rules.
- **Isolation Forest:** identifies anomalies by fitting an anomaly-detection model to the uploaded data.
- **Hybrid:** combines rule-based and Isolation Forest alerts.

Isolation Forest is fitted to each uploaded dataset. Its predictions should not be interpreted as the output of a separately trained production model.

## Offline Research Results

The dashboard loads precomputed research summaries from:

- `research/results/final_model_comparison.csv`
- `research/results/final_ann_test_thresholds.csv`

These results are separate from live CSV analysis. They represent offline experiments and should not be interpreted as current production performance.

## Run Automated Tests

Use a separate SQLite database for testing. In Windows PowerShell:

```powershell
$env:DATABASE_URL = "sqlite:///detectbench_test.db"
python -m pytest -v
Remove-Item Env:DATABASE_URL
```

Do not point the test database variable at the production database.

## Database Migrations

Database schema changes are managed using Flask-Migrate.

```powershell
flask db upgrade
flask db current
```

Do not manually modify the production database schema or run migrations against an unintended database.

## Production Deployment

The deployed Render service uses PostgreSQL through `DATABASE_URL`.

Configure these environment variables in the hosting dashboard:

- `SECRET_KEY` — a strong, private secret
- `DATABASE_URL` — the production PostgreSQL connection string

Do not publish actual secret values in this repository.

The configured Render start command is:

```bash
flask db upgrade && gunicorn app:app
```

## Security Notes

- Authentication is required for protected dashboard, history, and report routes.
- Passwords are stored as password hashes.
- Analysis history is associated with the logged-in user.
- CSRF protection is enabled.
- Upload size and CSV structure are validated.
- Never commit `.env` files, credentials, raw private datasets, or production database URLs.

## Limitations

- Detection quality depends on the input data and the selected detection method.
- Precision, Recall, and F1-score require valid `actual_label` values.
- Offline research metrics and live uploaded-data results are distinct.
- This project is a defensive security analysis prototype and should be evaluated further before use in a real production security environment.

## Repository

GitHub: https://github.com/avadhutchougoule3862/DetectBench-AI

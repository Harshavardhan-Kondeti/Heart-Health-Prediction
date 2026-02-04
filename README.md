# Heart Health Web App (FastAPI)

End-to-end web application for ECG/PPG/heart-risk analysis. Users can register/login, upload ECG images or CSVs, run ML inference (CNN for ECG images, LightGBM for heart risk, sklearn model for PPG), view dashboards, and download PDF reports. Built with FastAPI, SQLAlchemy, Jinja2, and model runtimes (TensorFlow-CPU, LightGBM, scikit-learn).

## Features
- Authentication with email verification, login, reset
- Uploads: ECG images (JPEG/PNG) and heart CSV; PPG uploads supported
- ML inference: CNN image classifier, heart-disease LightGBM model, PPG health model
- Fusion reports: combined outputs with PDF generation
- Dashboard: history, normal/abnormal stats
- Persistent storage for uploads and reports

## Repository layout
```
├─ backend/
│  ├─ main.py                 # FastAPI app entry
│  ├─ auth.py                 # Auth helpers
│  ├─ database.py             # SQLAlchemy engine/session/init
│  ├─ models.py               # ORM models (User, Submission, etc.)
│  ├─ schemas.py              # Pydantic schemas
│  ├─ routers/                # Route modules (auth, upload, ppg, heart, fusion, report)
│  ├─ ml/
│  │  ├─ inference.py         # Model loading and prediction
│  │  └─ preprocess.py        # Signal/image preprocessing
│  └─ templates/              # Jinja2 templates (UI)
├─ models/                    # Model artifacts (local dev only)
├─ uploads/                   # Uploaded files (runtime)
├─ reports/                   # Generated PDFs (runtime)
├─ requirements.txt
└─ README.md
```

## Requirements
- Python 3.11 (recommended)
- pip
- For local inference: CPU with enough RAM for TensorFlow and LightGBM

## Environment variables
Create `.env` at repo root (loaded by `python-dotenv`). Common keys:
- `DATABASE_URL` (defaults to `sqlite:///./ecg_app.db` locally)
- `SECRET_KEY` or JWT/email-related secrets if used in `auth.py`
- `MODEL_DIR` (optional; default `models`) for model artifact directory

Example `.env`:
```
DATABASE_URL=sqlite:///./ecg_app.db
MODEL_DIR=models
```

## Install and run locally (Windows PowerShell)
```powershell
cd "C:\Users\91900\Desktop\Cursor AI\Project"
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# optional: set env vars for this session
$env:DATABASE_URL="sqlite:///./ecg_app.db"
$env:MODEL_DIR="models"

# ensure runtime folders exist
mkdir uploads 2>$null
mkdir reports 2>$null

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000`.

## Data flows and models
- ECG image classification: loads `models/ecg_cnn_effb0.keras` (TensorFlow CPU). Upload images via UI; prediction stored with submission.
- Heart disease risk: uses LightGBM model and scaler files under `models/HeartDiseaseModel/`.
- PPG health: uses sklearn model and PCA transformer under `models/PPG_Model/`.

Set `MODEL_DIR` to point to where these artifacts live. In production on Render, prefer object storage + local cache.

### File conventions
- ECG CNN: `ecg_cnn_effb0.keras`
- Heart: `heart_disease_model_LGBM_2022.pkl`, `heart_disease_scaler_LGBM_2022.pkl`
- PPG: `ppg_health_model.pkl`, `ppg_pca_transformer.pkl`

## Deploy to Railway.app

Railway is recommended for this project.

### Steps:
1.  **Sign up** at [railway.app](https://railway.app).
2.  **New Project** → **Deploy from GitHub repo**.
3.  **Add Database**: Add a PostgreSQL service.
4.  **Configure Variables**:
    *   Set `DATABASE_URL` by referencing the PostgreSQL service variable.
    *   Set `MODEL_DIR=models`.
5.  **Models**: Ensure your ML models are available (either in git or downloaded via a script).

### Notes for production
- Use PostgreSQL (required for persistence across restarts).
- Filesystem is ephemeral unless using external storage (S3/R2).
- Ensure instance has enough RAM for ML models (≥1-2 GB).

## Common operations
- Run server locally: `uvicorn backend.main:app --reload`
- Lint/type check: ensure your editor runs them; no separate config required here.
- Database init: handled at startup via `init_db()` in `backend/database.py` (adds missing columns if absent).

## Troubleshooting
- Cannot download models on Windows with `curl -L`: use `curl.exe` or `Invoke-WebRequest`.
- On Render, paths are Linux: project root is `/opt/render/project/src`.
- Large model errors during build: move models to object storage and download at runtime.

## Security/Secrets
- Never commit secrets. Set them in Render Environment.
- Rotate keys if exposed.

## License
Proprietary or add your preferred license.

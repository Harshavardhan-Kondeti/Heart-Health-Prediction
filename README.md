# ECG Health Prediction Web App (FastAPI)

A minimal web application for patients to register/login, enter details, upload ECG signals, and run inference using your pre-trained ML model.

## Features
- JWT-based authentication (register/login)
- Patient submission form (age, sex, notes) + ECG file upload
- Server-side preprocessing (CSV or NumPy) and model inference stub
- Pluggable model loader (sklearn pickle or ONNX)
- SQLite database for users and submissions
- Jinja templates for simple UI

## Project Structure
```
Project/
├─ requirements.txt
├─ README.md
├─ .env.example
├─ backend/
│  ├─ main.py
│  ├─ auth.py
│  ├─ database.py
│  ├─ schemas.py
│  ├─ models.py
│  ├─ ml/
│  │  ├─ inference.py
│  │  └─ preprocess.py
│  └─ templates/
│     ├─ base.html
│     ├─ login.html
│     ├─ register.html
│     ├─ dashboard.html
│     └─ upload.html
└─ models/
   └─ model.pkl (or model.onnx)
```

## Prerequisites
- Python 3.10+
- Windows PowerShell or terminal

## Setup (Windows PowerShell)
```powershell
# From the project root
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Copy env template and edit secrets
copy .env.example .env
# Edit .env to set SECRET_KEY (any long random string)

# Create folders for model if not present
mkdir models 2>$null

# Run the app
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

## Exporting Your Colab Model
Export your trained model from Colab in one of these ways and place it in `models/`:
- Sklearn/XGBoost/etc. pickle:
  ```python
  import joblib
  joblib.dump(trained_model, 'model.pkl')
  ```
- ONNX (recommended for portability):
  ```python
  import skl2onnx
  # convert your pipeline or estimator to ONNX
  # save as model.onnx
  ```

The backend will attempt to load `models/model.onnx` first, then `models/model.pkl`.

## File Upload Format
- CSV: one column of ECG samples, with or without header
- NumPy: `.npy` 1D or 2D array; if 2D, first column is used

## Notes
- This app is a starter. Adjust preprocessing in `backend/ml/preprocess.py` to match your training pipeline.
- Replace `predict_proba`/`predict` logic in `backend/ml/inference.py` to reflect your model’s API.
- For production, use a persistent database and configure HTTPS, CORS, and secrets management.

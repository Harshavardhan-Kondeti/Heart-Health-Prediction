import io
import os
from typing import Optional
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Submission


router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")


_this_file = Path(__file__).resolve()
_project_root = _this_file.parents[2]  # <repo_root>
_default_model_dir = _project_root / "models" / "HeartDiseaseModel"

MODEL_DIR = Path(os.getenv("HEART_MODEL_DIR", str(_default_model_dir))).resolve()
MODEL_PATH = str((MODEL_DIR / "heart_disease_model_LGBM_2022.pkl").resolve())
SCALER_PATH = str((MODEL_DIR / "heart_disease_scaler_LGBM_2022.pkl").resolve())


def _load_model_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Missing files at {MODEL_PATH} or {SCALER_PATH}"
        )
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")
    return model, scaler


def _advise_for_probability(prob: float) -> str:
    if prob < 0.2:
        return (
            "Low predicted risk. Maintain healthy weight, exercise 150 min/week, balanced diet, "
            "avoid tobacco, and monitor BP/lipids annually."
        )
    if prob < 0.5:
        return (
            "Moderate predicted risk. Consider clinician review, check BP, fasting lipids, HbA1c. "
            "Adopt Mediterranean diet, reduce sodium, and increase physical activity."
        )
    if prob < 0.8:
        return (
            "Elevated predicted risk. Schedule appointment for cardiovascular evaluation, "
            "optimize BP/lipids/diabetes control, consider statin eligibility assessment."
        )
    return (
        "High predicted risk. Seek prompt clinical assessment. Discuss statins/antiplatelets as indicated, "
        "evaluate for symptoms (chest pain, dyspnea). If acute symptoms, seek urgent care."
    )


@router.get("/heart/upload", response_class=HTMLResponse)
async def heart_csv_upload_page(request: Request, user=Depends(get_current_user)):
    return templates.TemplateResponse("heart_csv_upload.html", {"request": request, "user": user})


@router.post("/heart/upload")
async def heart_csv_upload(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".csv"):
        return templates.TemplateResponse(
            "heart_csv_upload.html",
            {"request": request, "user": user, "error": "Please upload a .csv file."},
        )

    data = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception:
        return templates.TemplateResponse(
            "heart_csv_upload.html",
            {"request": request, "user": user, "error": "Failed to parse CSV. Verify formatting."},
        )

    # Optional true labels
    y_true = None
    if "HadHeartAttack" in df.columns:
        y_true = df["HadHeartAttack"].map({"Yes": 1, "No": 0})
        df = df.drop(columns=["HadHeartAttack"])  # ensure model columns only

    # Basic imputation mirroring training/testing approach
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            except Exception:
                df[col] = df[col].fillna("")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Label-encode object columns (fit-on-input to mirror provided testing script)
    from sklearn.preprocessing import LabelEncoder  # local import to avoid global dependency issues

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Load model and scaler
    try:
        model, scaler = _load_model_scaler()
    except FileNotFoundError as e:
        return templates.TemplateResponse(
            "heart_csv_upload.html",
            {
                "request": request,
                "user": user,
                "error": (
                    f"Model/scaler not found. Checked paths: {MODEL_PATH} and {SCALER_PATH}. "
                    "Set HEART_MODEL_DIR if stored elsewhere."
                ),
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "heart_csv_upload.html",
            {
                "request": request,
                "user": user,
                "error": f"Failed to load model/scaler due to dependency or format issue: {e}",
            },
        )

    # Scale and predict
    X_scaled = scaler.transform(df.values)
    y_pred = model.predict(X_scaled)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_scaled)[:, 1]
    except Exception:
        pass

    # Prepare outputs
    predictions = ["Abnormal (Heart Disease)" if p == 1 else "Normal" for p in y_pred]
    probs = [float(p) for p in (y_prob if y_prob is not None else np.zeros(len(y_pred)))]
    advices = [_advise_for_probability(p) for p in probs]

    # Persist one submission summary (first row) for dashboard consistency
    top_prob = float(probs[0]) if len(probs) else 0.0
    top_label = "Abnormal" if (len(y_pred) and y_pred[0] == 1) else "Normal"

    # Save uploaded CSV to disk
    uploads_dir = os.path.join("uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    out_name = f"{ts}_{file.filename}"
    out_path = os.path.join(uploads_dir, out_name)
    with open(out_path, "wb") as f:
        f.write(data)

    submission = Submission(
        user_id=user.id,
        test_type="HEART_CSV",
        file_name=out_path,
        predicted_label=top_label,
        predicted_score=str(top_prob),
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    # Build a compact preview table (first 10 predictions)
    preview_rows = min(10, len(predictions))
    table = [
        {"index": i, "prediction": predictions[i], "prob": probs[i], "advice": advices[i]}
        for i in range(preview_rows)
    ]

    return templates.TemplateResponse(
        "heart_csv_upload.html",
        {
            "request": request,
            "user": user,
            "table": table,
            "count": len(predictions),
            "download_msg": "Processed successfully. Predictions preview shown below.",
        },
    )



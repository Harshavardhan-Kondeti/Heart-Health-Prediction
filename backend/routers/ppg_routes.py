import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import numpy as np

from ..auth import get_current_user
from ..database import get_db
from ..models import Submission
from ..ml.inference import PPGImageService
from ..ml.preprocess import load_image_from_bytes


router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")
ppg_service = PPGImageService()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _ppg_advice_text(binary: Optional[str]) -> str:
    if binary == "Normal":
        return (
            "Prediction suggests a normal PPG. Maintain healthy lifestyle; monitor if symptoms occur."
        )
    return (
        "Prediction suggests possible abnormality (MI risk). Please consult a cardiologist promptly."
    )


@router.get("/upload", response_class=HTMLResponse)
async def ppg_upload_page(request: Request, user=Depends(get_current_user)):
    return templates.TemplateResponse("ppg_upload.html", {"request": request, "user": user})


@router.post("/upload")
async def ppg_upload(
    request: Request,
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Read bytes for both saving and processing
    data = await file.read()

    # Save to disk with timestamp prefix
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    safe_name = f"{ts}_{file.filename}"
    out_path = UPLOAD_DIR / safe_name
    with out_path.open("wb") as f:
        f.write(data)

    # Load image array (RGB) then predict via PPG service
    img_arr = await load_image_from_bytes(data, file.filename or "")
    if img_arr.size == 0:
        result = {"warning": "Unsupported file type. Please upload a PNG/JPG image of the PPG signal."}
    else:
        result = ppg_service.predict_from_image_array(img_arr)

    # Save record
    submission = Submission(
        user_id=user.id,
        test_type="PPG",
        age=age,
        sex=sex,
        notes=notes,
        file_name=str(out_path),
        predicted_label=str(result.get("label")) if result.get("label") is not None else None,
        predicted_score=str(result.get("score")) if result.get("score") is not None else None,
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    advice = _ppg_advice_text(result.get("binary"))

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "result": result,
            "submission": submission,
            "advice": advice,
        },
    )



import os
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path

from ..auth import get_current_user
from ..database import get_db
from ..models import Submission
from ..ml.inference import ModelService
from ..ml.preprocess import load_input_from_bytes

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")
model_service = ModelService()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _advice_text(binary: Optional[str]) -> str:
    if binary == "Normal":
        return (
            "Prediction suggests a normal ECG. If you have symptoms, consider a routine check-up. "
            "General advice: maintain healthy lifestyle, periodic 12‑lead ECG if clinically indicated."
        )
    return (
        "Prediction suggests possible abnormality. Please consult a cardiologist. Suggested tests: "
        "12‑lead ECG, cardiac troponin (if acute symptoms), echocardiogram, Holter monitoring, "
        "and/or exercise stress test as per clinical judgment. If chest pain or severe symptoms, seek urgent care."
    )


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request, user=Depends(get_current_user)):
    return templates.TemplateResponse("upload.html", {"request": request, "user": user})


@router.post("/upload")
async def upload(
    request: Request,
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    img_size = model_service.get_expected_image_size()

    # Read bytes (so we can both parse and save)
    data = await file.read()

    # Save to disk with timestamp prefix
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    safe_name = f"{ts}_{file.filename}"
    out_path = UPLOAD_DIR / safe_name
    with out_path.open("wb") as f:
        f.write(data)

    # Load input array
    arr = await load_input_from_bytes(data, file.filename or "", image_size=img_size)

    # Predict
    result = model_service.predict(arr)

    # Save record
    submission = Submission(
        user_id=user.id,
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

    advice = _advice_text(result.get("binary"))

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

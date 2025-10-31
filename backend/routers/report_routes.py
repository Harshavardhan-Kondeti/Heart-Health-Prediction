from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import smtplib
from email.message import EmailMessage

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy.orm import Session
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from ..auth import get_current_user
from ..database import get_db
from ..models import Submission, User

router = APIRouter()
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _generate_pdf(path: Path, user: User, sub: Submission, result: dict, advice: Optional[str]) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Heart Health Report")
    y -= 25

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient Details")
    y -= 18
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Name: {user.full_name or ''}")
    y -= 16
    c.drawString(50, y, f"Email: {user.email}")
    y -= 16
    c.drawString(50, y, f"Age: {sub.age or ''}  Sex: {sub.sex or ''}")
    y -= 16

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Prediction")
    y -= 18
    c.setFont("Helvetica", 11)
    label = result.get("top_label") or result.get("label")
    c.drawString(50, y, f"Top class: {label}")
    y -= 16
    if result.get("binary"):
        c.drawString(50, y, f"Normal/Abnormal: {result.get('binary')}")
        y -= 16
    if result.get("score") is not None:
        c.drawString(50, y, f"Score: {result.get('score')}")
        y -= 16

    if advice:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Advice")
        y -= 18
        c.setFont("Helvetica", 11)
        for line in [advice[i:i+100] for i in range(0, len(advice), 100)]:
            c.drawString(50, y, line)
            y -= 14

    try:
        if sub.file_name and any(sub.file_name.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg"]):
            img_reader = ImageReader(sub.file_name)
            img_w, img_h = img_reader.getSize()
            max_w = width - 100
            scale = min(max_w / img_w, 300 / img_h)
            draw_w, draw_h = img_w * scale, img_h * scale
            c.drawImage(img_reader, 50, max(50, y - draw_h - 10), width=draw_w, height=draw_h, preserveAspectRatio=True, anchor='sw')
    except Exception:
        pass

    c.showPage()
    c.save()


def _smtp_config():
    return {
        "host": os.getenv("SMTP_HOST", ""),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"),
        "sender": os.getenv("SMTP_SENDER", os.getenv("SMTP_USER", "")),
    }


def _send_email_with_attachment(to_email: str, subject: str, body: str, attachment_path: Path) -> None:
    cfg = _smtp_config()
    if not cfg["host"] or not cfg["sender"]:
        raise RuntimeError("SMTP not configured. Set SMTP_HOST, SMTP_USER/SMTP_SENDER, SMTP_PASSWORD.")

    msg = EmailMessage()
    msg["From"] = cfg["sender"]
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        data = f.read()
    msg.add_attachment(data, maintype="application", subtype="pdf", filename=attachment_path.name)

    server = smtplib.SMTP(cfg["host"], cfg["port"])
    if cfg["use_tls"]:
        server.starttls()
    if cfg["user"]:
        server.login(cfg["user"], cfg["password"])  # may be no-op if empty
    server.send_message(msg)
    server.quit()


@router.get("/report/{submission_id}")
async def download_report(submission_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    sub = db.query(Submission).filter(Submission.id == submission_id, Submission.user_id == user.id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    result = {
        "label": sub.predicted_label,
        "score": sub.predicted_score,
    }
    if sub.predicted_label:
        if isinstance(sub.predicted_label, str) and "normal" in sub.predicted_label.lower():
            result["binary"] = "Normal"
        else:
            try:
                idx = int(sub.predicted_label)
                result["binary"] = "Normal" if idx == 3 else "Abnormal"
            except Exception:
                result["binary"] = "Abnormal"

    advice = (
        "Prediction suggests a normal ECG. If you have symptoms, consider a routine check-up. "
        "General advice: maintain healthy lifestyle, periodic 12‑lead ECG if clinically indicated."
        if result.get("binary") == "Normal" else
        "Prediction suggests possible abnormality. Please consult a cardiologist. Suggested tests: "
        "12‑lead ECG, cardiac troponin (if acute symptoms), echocardiogram, Holter monitoring, "
        "and/or exercise stress test as per clinical judgment. If chest pain or severe symptoms, seek urgent care."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"report_{submission_id}.pdf"
    _generate_pdf(out_path, user, sub, result, advice)
    return FileResponse(str(out_path), media_type="application/pdf", filename=out_path.name)


@router.get("/report/{submission_id}/email")
async def email_report(submission_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    sub = db.query(Submission).filter(Submission.id == submission_id, Submission.user_id == user.id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    result = {
        "label": sub.predicted_label,
        "score": sub.predicted_score,
    }
    if sub.predicted_label:
        if isinstance(sub.predicted_label, str) and "normal" in sub.predicted_label.lower():
            result["binary"] = "Normal"
        else:
            try:
                idx = int(sub.predicted_label)
                result["binary"] = "Normal" if idx == 3 else "Abnormal"
            except Exception:
                result["binary"] = "Abnormal"

    advice = (
        "Prediction suggests a normal ECG. If you have symptoms, consider a routine check-up. "
        "General advice: maintain healthy lifestyle, periodic 12‑lead ECG if clinically indicated."
        if result.get("binary") == "Normal" else
        "Prediction suggests possible abnormality. Please consult a cardiologist. Suggested tests: "
        "12‑lead ECG, cardiac troponin (if acute symptoms), echocardiogram, Holter monitoring, "
        "and/or exercise stress test as per clinical judgment. If chest pain or severe symptoms, seek urgent care."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"report_{submission_id}.pdf"
    _generate_pdf(out_path, user, sub, result, advice)

    # Compose a friendly email body
    patient_name = user.full_name or user.email
    lines = [
        f"Dear {patient_name},",
        "",
        "Thank you for using the Heart Health Assistant. Please find attached your ECG prediction report.",
        "",
        f"Summary:",
        f"  - Result: {result.get('binary','N/A')}",
        f"  - Score: {result.get('score','N/A')}",
        f"  - Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "Advice:",
        f"  {advice}",
        "",
        "This report is provided for informational triage and is not a medical diagnosis.",
        "Please consult a qualified clinician for medical advice.",
        "",
        "Best regards,",
        "Heart Health Assistant",
        "",
        "Note: This is an automated message. Please do not reply to this email.",
    ]
    body = "\n".join(lines)

    try:
        _send_email_with_attachment(
            to_email=user.email,
            subject="Your ECG Prediction Report",
            body=body,
            attachment_path=out_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

    # Redirect with acknowledgement flag
    return RedirectResponse(url="/dashboard?email=sent", status_code=303)

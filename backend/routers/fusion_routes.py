from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import os
import smtplib
from email.message import EmailMessage
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Submission


router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_score(score_str: Optional[str]) -> Optional[float]:
    if score_str is None:
        return None
    try:
        return float(score_str)
    except Exception:
        return None


def _label_to_risk(label: Optional[str]) -> Optional[float]:
    if not label:
        return None
    try:
        if label.isdigit():
            # legacy numeric class index, assume normal when idx==3
            return 0.0 if int(label) == 3 else 1.0
    except Exception:
        pass
    s = label.lower()
    if "normal" in s:
        return 0.0
    return 1.0


def _normalize_score(score: Optional[float], label: Optional[str]) -> float:
    # Use probability if available; otherwise map label to 0/1
    if score is not None:
        # Clip to [0,1]
        if score < 0.0:
            return 0.0
        if score > 1.0:
            # Some models may output logit-like; squash simple heuristic
            try:
                return 1.0 if score >= 0.5 else 0.0
            except Exception:
                return 1.0
        return score
    mapped = _label_to_risk(label)
    return mapped if mapped is not None else 0.5


def _aggregate_fusion(scores: Dict[str, float]) -> Tuple[float, str]:
    # Weights by modality; tweakable
    weights = {
        "ECG": 0.45,
        "PPG": 0.25,
        "HEART_CSV": 0.30,
    }
    total_w = 0.0
    total = 0.0
    for k, v in scores.items():
        w = weights.get(k, 0.2)
        total += w * v
        total_w += w
    fused = total / total_w if total_w > 0 else 0.5

    if fused < 0.2:
        status = "Low risk"
    elif fused < 0.5:
        status = "Moderate risk"
    elif fused < 0.8:
        status = "Elevated risk"
    else:
        status = "High risk"
    return fused, status


def _build_guidance(fused: float) -> Dict[str, List[str]]:
    # Tailored recommendations by risk tier
    if fused < 0.2:
        precautions = [
            "Maintain routine annual checkups and blood pressure/lipid screening.",
            "Continue 150 minutes/week of moderate aerobic exercise.",
            "Avoid tobacco exposure and maintain healthy BMI.",
        ]
        measurements = [
            "Home BP: weekly if history of hypertension, otherwise monthly.",
            "Weight and waist circumference monthly.",
        ]
        consult = [
            "Primary care physician for preventive care as needed.",
        ]
    elif fused < 0.5:
        precautions = [
            "Increase physical activity; aim for 150–300 minutes/week.",
            "Adopt Mediterranean-style diet; limit sodium to <2g/day.",
            "Target sleep 7–9 hours; manage stress with mindfulness.",
        ]
        measurements = [
            "Check BP twice weekly for 2 weeks; track average.",
            "Fasting lipids and HbA1c at next visit if not done in 12 months.",
        ]
        consult = [
            "Primary care for risk review; consider statin eligibility per guidelines.",
        ]
    elif fused < 0.8:
        precautions = [
            "Prioritize BP, glucose, and lipid control; adhere to medications if prescribed.",
            "Reduce refined carbs and saturated fats; increase fiber and omega‑3 sources.",
            "Avoid smoking/vaping; limit alcohol to recommended amounts.",
        ]
        measurements = [
            "Home BP daily for 1–2 weeks; bring log to clinic.",
            "Consider ambulatory BP monitoring if variability is high.",
        ]
        consult = [
            "Schedule a clinician visit for cardiovascular risk optimization.",
            "Discuss need for echocardiogram or stress testing based on symptoms and history.",
        ]
    else:
        precautions = [
            "Seek prompt clinical assessment, especially if chest pain, dyspnea, or syncope.",
            "Avoid strenuous exertion until cleared by clinician.",
            "Strict adherence to cardio‑protective medications if prescribed.",
        ]
        measurements = [
            "Immediate BP/HR assessment; track vitals if advised.",
            "If acute symptoms, urgent evaluation including ECG and troponin per clinician.",
        ]
        consult = [
            "Cardiologist consultation recommended.",
            "Emergency care if acute concerning symptoms occur.",
        ]

    diet = [
        "Emphasize vegetables, fruits, legumes, nuts, whole grains.",
        "Prefer olive oil; limit processed foods, trans fats, and high sodium.",
        "Fish 2x/week; consider plant‑based proteins routinely.",
    ]
    habits = [
        "No smoking or vaping; avoid secondhand smoke.",
        "Regular physical activity; incorporate strength training 2x/week.",
        "Sleep hygiene: consistent schedule, dark/cool room, limit screens before bed.",
    ]

    return {
        "precautions": precautions,
        "measurements": measurements,
        "consult": consult,
        "diet": diet,
        "habits": habits,
    }


@router.get("/fusion/report", response_class=HTMLResponse)
async def fusion_report(request: Request, user=Depends(get_current_user), db: Session = Depends(get_db)):
    subs: List[Submission] = (
        db.query(Submission)
        .filter(Submission.user_id == user.id)
        .order_by(Submission.created_at.desc())
        .all()
    )
    if not subs:
        return templates.TemplateResponse(
            "fusion_report.html",
            {"request": request, "user": user, "error": "No submissions found. Upload tests to generate a report."},
        )

    # Take the most recent per test type
    latest_by_type: Dict[str, Submission] = {}
    for s in subs:
        t = (s.test_type or "").upper() or "ECG"
        if t not in latest_by_type:
            latest_by_type[t] = s

    # Normalize to risk scores
    modality_scores: Dict[str, float] = {}
    details: List[Dict[str, str]] = []
    for t, s in latest_by_type.items():
        score = _parse_score(s.predicted_score)
        risk = _normalize_score(score, s.predicted_label)
        modality_scores[t] = risk
        details.append({
            "type": t,
            "result": ("Normal" if risk < 0.5 else "Abnormal"),
            "score": f"{risk:.3f}",
            "source": s.file_name or "",
            "time": s.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        })

    fused, status = _aggregate_fusion(modality_scores)
    guidance = _build_guidance(fused)

    return templates.TemplateResponse(
        "fusion_report.html",
        {
            "request": request,
            "user": user,
            "fused": f"{fused:.3f}",
            "status": status,
            "modalities": details,
            "guidance": guidance,
        },
    )


def _smtp_config():
    return {
        "host": os.getenv("SMTP_HOST", ""),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"),
        "sender": os.getenv("SMTP_SENDER", os.getenv("SMTP_USER", "")),
    }


def _generate_fusion_pdf(path: Path, user, fused: float, status: str, modalities: List[Dict[str, str]], guidance: Dict[str, List[str]]):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as e:
        raise RuntimeError("reportlab is required to generate PDF")

    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Overall Heart Health Report")
    y -= 25

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y -= 18

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Patient")
    y -= 16
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Name: {getattr(user, 'full_name', '') or ''}")
    y -= 14
    c.drawString(50, y, f"Email: {getattr(user, 'email', '')}")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Fusion Summary")
    y -= 16
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Overall Risk Score: {fused:.3f}")
    y -= 14
    c.drawString(50, y, f"Status: {status}")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Inputs Considered")
    y -= 16
    c.setFont("Helvetica", 10)
    for m in modalities:
        line = f"- {m.get('type','')}: {m.get('result','')} (score {m.get('score','')}) at {m.get('time','')}"
        c.drawString(50, y, line)
        y -= 14
        if y < 80:
            c.showPage(); y = height - 50

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Guidance")
    y -= 16
    c.setFont("Helvetica", 10)
    for section in ("Precautions", "Measurements", "Consult", "Diet", "Habits"):
        items = guidance.get(section.lower(), []) if isinstance(guidance, dict) else []
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, section)
        y -= 14
        c.setFont("Helvetica", 10)
        for g in items:
            for chunk in [g[i:i+95] for i in range(0, len(g), 95)]:
                c.drawString(60, y, f"• {chunk}")
                y -= 12
                if y < 80:
                    c.showPage(); y = height - 50
        y -= 8

    c.showPage()
    c.save()


@router.get("/fusion/report/pdf")
async def fusion_report_pdf(user=Depends(get_current_user), db: Session = Depends(get_db)):
    # Recompute fusion to ensure latest data
    subs: List[Submission] = (
        db.query(Submission)
        .filter(Submission.user_id == user.id)
        .order_by(Submission.created_at.desc())
        .all()
    )
    if not subs:
        raise HTTPException(status_code=404, detail="No submissions found")

    latest_by_type: Dict[str, Submission] = {}
    for s in subs:
        t = (s.test_type or "").upper() or "ECG"
        if t not in latest_by_type:
            latest_by_type[t] = s

    modality_scores: Dict[str, float] = {}
    details: List[Dict[str, str]] = []
    for t, s in latest_by_type.items():
        score = _parse_score(s.predicted_score)
        risk = _normalize_score(score, s.predicted_label)
        modality_scores[t] = risk
        details.append({
            "type": t,
            "result": ("Normal" if risk < 0.5 else "Abnormal"),
            "score": f"{risk:.3f}",
            "source": s.file_name or "",
            "time": s.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        })

    fused, status = _aggregate_fusion(modality_scores)
    guidance = _build_guidance(fused)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"fusion_{user.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
    _generate_fusion_pdf(out_path, user, fused, status, details, guidance)
    return FileResponse(str(out_path), media_type="application/pdf", filename=out_path.name)


@router.get("/fusion/report/email")
async def fusion_report_email(user=Depends(get_current_user), db: Session = Depends(get_db)):
    # Generate PDF first
    subs: List[Submission] = (
        db.query(Submission)
        .filter(Submission.user_id == user.id)
        .order_by(Submission.created_at.desc())
        .all()
    )
    if not subs:
        raise HTTPException(status_code=404, detail="No submissions found")

    latest_by_type: Dict[str, Submission] = {}
    for s in subs:
        t = (s.test_type or "").upper() or "ECG"
        if t not in latest_by_type:
            latest_by_type[t] = s

    modality_scores: Dict[str, float] = {}
    details: List[Dict[str, str]] = []
    for t, s in latest_by_type.items():
        score = _parse_score(s.predicted_score)
        risk = _normalize_score(score, s.predicted_label)
        modality_scores[t] = risk
        details.append({
            "type": t,
            "result": ("Normal" if risk < 0.5 else "Abnormal"),
            "score": f"{risk:.3f}",
            "source": s.file_name or "",
            "time": s.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        })

    fused, status = _aggregate_fusion(modality_scores)
    guidance = _build_guidance(fused)
    out_path = REPORTS_DIR / f"fusion_{user.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
    _generate_fusion_pdf(out_path, user, fused, status, details, guidance)

    cfg = _smtp_config()
    if not cfg["host"] or not cfg["sender"]:
        raise HTTPException(status_code=500, detail="SMTP not configured. Set SMTP_HOST, SMTP_USER/SMTP_SENDER, SMTP_PASSWORD.")

    msg = EmailMessage()
    msg["From"] = cfg["sender"]
    msg["To"] = user.email
    msg["Subject"] = "Your Overall Heart Health Report"
    body_lines = [
        f"Dear {getattr(user, 'full_name', '') or user.email},",
        "",
        "Please find attached your overall heart health fusion report.",
        "",
        f"Overall Risk Score: {fused:.3f}",
        f"Status: {status}",
        "",
        "This report is for informational purposes and not a medical diagnosis.",
        "Please consult a licensed clinician for medical advice.",
    ]
    msg.set_content("\n".join(body_lines))
    with open(out_path, "rb") as f:
        data = f.read()
    msg.add_attachment(data, maintype="application", subtype="pdf", filename=out_path.name)

    try:
        server = smtplib.SMTP(cfg["host"], cfg["port"])
        if cfg["use_tls"]:
            server.starttls()
        if cfg["user"]:
            server.login(cfg["user"], cfg["password"])
        server.send_message(msg)
        server.quit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

    return RedirectResponse(url="/fusion/report?email=sent", status_code=303)



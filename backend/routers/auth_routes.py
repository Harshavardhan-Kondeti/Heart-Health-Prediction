from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, UserProfile
from ..schemas import UserCreate, Token
from ..auth import get_password_hash, authenticate_user, create_access_token

# new imports
from pathlib import Path
import csv
import re
import os
import smtplib
import random
from email.message import EmailMessage
from jose import jwt

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")
DATA_DIR = Path("data")
USERS_CSV = DATA_DIR / "users.csv"


def _profile_complete(db: Session, user_id: int) -> bool:
    prof = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    return bool(prof and prof.phone and prof.guardian_name and prof.guardian_phone and prof.guardian_email)


def _smtp_config():
    return {
        "host": os.getenv("SMTP_HOST", ""),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes"),
        "sender": os.getenv("SMTP_SENDER", os.getenv("SMTP_USER", "")),
    }


def _send_email(to_email: str, subject: str, body: str) -> None:
    cfg = _smtp_config()
    if not cfg["host"] or not cfg["sender"]:
        raise RuntimeError("SMTP not configured. Set SMTP_HOST, SMTP_USER/SMTP_SENDER, SMTP_PASSWORD.")
    msg = EmailMessage()
    msg["From"] = cfg["sender"]
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    server = smtplib.SMTP(cfg["host"], cfg["port"])
    if cfg["use_tls"]:
        server.starttls()
    if cfg["user"]:
        server.login(cfg["user"], cfg["password"])  # may be no-op if empty
    server.send_message(msg)
    server.quit()


def _make_token(email: str, purpose: str, minutes: int = 60) -> str:
    secret = os.getenv("SECRET_KEY", "change_this")
    alg = os.getenv("ALGORITHM", "HS256")
    exp = datetime.utcnow() + timedelta(minutes=minutes)
    payload = {"sub": email, "purpose": purpose, "exp": exp}
    return jwt.encode(payload, secret, algorithm=alg)


def _verify_token(token: str, expected_purpose: str) -> str | None:
    try:
        secret = os.getenv("SECRET_KEY", "change_this")
        alg = os.getenv("ALGORITHM", "HS256")
        data = jwt.decode(token, secret, algorithms=[alg])
        if data.get("purpose") != expected_purpose:
            return None
        return data.get("sub")
    except Exception:
        return None


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        return RedirectResponse("/auth/login?error=invalid", status_code=status.HTTP_302_FOUND)
    # Enforce verification (treat missing field as verified for legacy rows)
    try:
        verified_val = getattr(user, "is_verified", True)
        if not bool(verified_val):
            return RedirectResponse("/auth/login?verify_required=1", status_code=status.HTTP_302_FOUND)
    except Exception:
        pass

    access_token_expires = timedelta(minutes=60)
    access_token = create_access_token(subject=user.email, expires_delta=access_token_expires)

    dest = "/profile" if not _profile_complete(db, user.id) else "/home"
    response = RedirectResponse(url=dest, status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="Authorization", value=f"Bearer {access_token}", httponly=False, samesite="lax")
    return response


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@router.post("/register")
async def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    full_name: str = Form("") ,
    db: Session = Depends(get_db),
):
    try:
        if len(password) < 5 or not re.search(r"[A-Z]", password) or not re.search(r"\d", password):
            return templates.TemplateResponse(
                "register.html",
                {"request": request, "error": "Password must be ≥5 chars, include one uppercase and one digit."},
            )

        existing = db.query(User).filter(User.email == email).first()
        if existing:
            return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered"})
        user = User(email=email, hashed_password=get_password_hash(password), full_name=full_name, is_verified=False)
        db.add(user)
        db.commit()
        db.refresh(user)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_exists = USERS_CSV.exists()
        with USERS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["id", "email", "full_name", "created_at"])
            writer.writerow([user.id, user.email, user.full_name or "", user.created_at.isoformat()])

        # Generate 6-digit verification code and email it
        try:
            code = f"{random.randint(0, 999999):06d}"
            user.verify_code = code
            user.verify_expires = datetime.utcnow() + timedelta(minutes=30)
            db.add(user)
            db.commit()

            pname = user.full_name or user.email
            body = "\n".join([
                f"Dear {pname},",
                "",
                "Welcome to Heart Health!",
                "Your email verification code is:",
                "",
                f"    {code}",
                "",
                "This code will expire in 30 minutes. If you did not sign up, please ignore this email.",
                "",
                "Best regards,",
                "Heart Health",
                "",
                "Note: This is an automated message. Please do not reply.",
            ])
            _send_email(user.email, "Heart Health – Your verification code", body)
        except Exception:
            pass

        # Redirect to code entry page
        return RedirectResponse(url=f"/auth/verify-code?email={user.email}", status_code=status.HTTP_302_FOUND)
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        return templates.TemplateResponse("register.html", {"request": request, "error": f"Registration failed: {str(e)}"})


@router.get("/verify-code", response_class=HTMLResponse)
async def verify_code_page(request: Request, email: str | None = None):
    return templates.TemplateResponse("verify_code.html", {"request": request, "email": email})


@router.post("/verify-code")
async def verify_code_submit(request: Request, email: str = Form(...), code: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return templates.TemplateResponse("verify_code.html", {"request": request, "email": email, "error": "User not found."})
    now = datetime.utcnow()
    if not user.verify_code or not user.verify_expires or user.verify_expires < now:
        return templates.TemplateResponse("verify_code.html", {"request": request, "email": email, "error": "Code expired. Please register again or request a new code."})
    if code.strip() != (user.verify_code or "").strip():
        return templates.TemplateResponse("verify_code.html", {"request": request, "email": email, "error": "Invalid code."})
    try:
        user.is_verified = True
        user.verify_code = None
        user.verify_expires = None
        db.add(user)
        db.commit()
    except Exception:
        db.rollback()
        return templates.TemplateResponse("verify_code.html", {"request": request, "email": email, "error": "Failed to verify. Try again."})
    return RedirectResponse(url="/auth/login?verified=1", status_code=303)


@router.get("/forgot", response_class=HTMLResponse)
async def forgot_page(request: Request):
    return templates.TemplateResponse("forgot.html", {"request": request})


@router.post("/forgot")
async def forgot_submit(request: Request, email: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if user:
        try:
            token = _make_token(user.email, purpose="reset", minutes=30)
            base = str(request.base_url).rstrip('/')
            link = f"{base}/auth/reset?token={token}"
            pname = user.full_name or user.email
            body = "\n".join([
                f"Dear {pname},",
                "",
                "We received a request to reset your password for the ECG Health Assistant.",
                "Click the link below to set a new password (valid for 30 minutes):",
                link,
                "",
                "If you did not request this, you can safely ignore this email.",
                "",
                "Best regards,",
                "ECG Health Assistant",
                "",
                "Note: This is an automated message. Please do not reply.",
            ])
            _send_email(user.email, "Reset your ECG Assistant password", body)
        except Exception:
            pass
    return RedirectResponse(url="/auth/login?reset_sent=1", status_code=303)


@router.get("/reset", response_class=HTMLResponse)
async def reset_page(request: Request, token: str):
    return templates.TemplateResponse("reset.html", {"request": request, "token": token})


@router.post("/reset")
async def reset_submit(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    email = _verify_token(token, expected_purpose="reset")
    if not email:
        return templates.TemplateResponse("reset.html", {"request": request, "token": token, "error": "Invalid or expired reset link."})
    if len(password) < 5 or not re.search(r"[A-Z]", password) or not re.search(r"\d", password):
        return templates.TemplateResponse("reset.html", {"request": request, "token": token, "error": "Password must be ≥5 chars, include one uppercase and one digit."})
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return templates.TemplateResponse("reset.html", {"request": request, "token": token, "error": "User not found."})
    user.hashed_password = get_password_hash(password)
    db.add(user)
    db.commit()
    return RedirectResponse(url="/auth/login?reset_ok=1", status_code=303)


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("Authorization")
    return response

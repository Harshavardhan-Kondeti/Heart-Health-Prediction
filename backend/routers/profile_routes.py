from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import UserProfile, Submission, User

router = APIRouter()
templates = Jinja2Templates(directory="backend/templates")


def profile_is_complete(profile: UserProfile | None) -> bool:
    if not profile:
        return False
    return bool(profile.phone and profile.guardian_name and profile.guardian_phone and profile.guardian_email)


@router.get("/profile", response_class=HTMLResponse)
async def get_profile(request: Request, user=Depends(get_current_user), db: Session = Depends(get_db)):
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    return templates.TemplateResponse("profile.html", {"request": request, "user": user, "profile": profile})


@router.post("/profile")
async def post_profile(
    request: Request,
    phone: str = Form(""),
    guardian_name: str = Form(""),
    guardian_phone: str = Form(""),
    guardian_email: str = Form(""),
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if not profile:
        profile = UserProfile(user_id=user.id)
        db.add(profile)
    profile.phone = phone.strip()
    profile.guardian_name = guardian_name.strip()
    profile.guardian_phone = guardian_phone.strip()
    profile.guardian_email = guardian_email.strip()
    db.commit()
    return RedirectResponse(url="/profile?saved=1", status_code=303)


@router.post("/profile/delete")
async def delete_account(user=Depends(get_current_user), db: Session = Depends(get_db)):
    # Delete submissions
    db.query(Submission).filter(Submission.user_id == user.id).delete(synchronize_session=False)
    # Delete profile
    db.query(UserProfile).filter(UserProfile.user_id == user.id).delete(synchronize_session=False)
    # Delete user
    db.query(User).filter(User.id == user.id).delete(synchronize_session=False)
    db.commit()
    resp = RedirectResponse(url="/auth/login?deleted=1", status_code=303)
    resp.delete_cookie("Authorization")
    return resp

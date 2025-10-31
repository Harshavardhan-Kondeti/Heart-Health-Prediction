from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserPublic(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str] = None

    class Config:
        from_attributes = True


class SubmissionCreate(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    notes: Optional[str] = None


class SubmissionPublic(BaseModel):
    id: int
    file_name: str
    predicted_label: Optional[str] = None
    predicted_score: Optional[str] = None

    class Config:
        from_attributes = True

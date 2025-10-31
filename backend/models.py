from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    submissions = relationship("Submission", back_populates="user")
    profile = relationship("UserProfile", back_populates="user", uselist=False)


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    phone = Column(String, nullable=True)
    guardian_name = Column(String, nullable=True)
    guardian_phone = Column(String, nullable=True)
    guardian_email = Column(String, nullable=True)

    user = relationship("User", back_populates="profile")


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    age = Column(Integer, nullable=True)
    sex = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    file_name = Column(String, nullable=False)
    predicted_label = Column(String, nullable=True)
    predicted_score = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="submissions")

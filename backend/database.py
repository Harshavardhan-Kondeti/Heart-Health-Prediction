import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ecg_app.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    # Lightweight migration: add users.is_verified if missing (SQLite)
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT 0"))
    except Exception:
        # Column likely exists; ignore
        pass
    # Add users.verify_code if missing
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN verify_code VARCHAR"))
    except Exception:
        pass
    # Add users.verify_expires if missing
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN verify_expires DATETIME"))
    except Exception:
        pass
    # Add submissions.test_type if missing
    try:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE submissions ADD COLUMN test_type VARCHAR"))
    except Exception:
        pass

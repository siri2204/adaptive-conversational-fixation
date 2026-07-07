"""
Persistence layer. SQLite by default (zero-setup for a course project);
swap DATABASE_URL for Postgres later without touching the rest of the app.

Tables:
  - sessions: one row per conversation
  - turns: every user/assistant message, with its embedding stored as JSON
  - intervention_events: every time a strategy fired (or was checked), with
    the fixation metrics at that point and what branches were offered/selected
"""
import json
from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    Boolean,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from app.config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ConversationSession(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    task_type = Column(String, nullable=False)  # e.g. story_generation, product_brainstorming, interface_design
    strategy = Column(String, nullable=False)  # static | fixed_interval | user_triggered | adaptive | baseline
    strategy_params = Column(Text, default="{}")  # JSON-encoded dict
    created_at = Column(DateTime, default=datetime.utcnow)
    last_intervention_turn = Column(Integer, default=-1)  # for cooldown tracking

    turns = relationship("Turn", back_populates="session", cascade="all, delete-orphan")
    events = relationship("InterventionEvent", back_populates="session", cascade="all, delete-orphan")

    def get_strategy_params(self) -> dict:
        return json.loads(self.strategy_params or "{}")


class Turn(Base):
    __tablename__ = "turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    role = Column(String, nullable=False)  # user | assistant | branch_selection
    content = Column(Text, nullable=False)
    embedding_json = Column(Text, nullable=False)  # JSON list[float]
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ConversationSession", back_populates="turns")

    def embedding(self) -> list[float]:
        return json.loads(self.embedding_json)


class InterventionEvent(Base):
    __tablename__ = "intervention_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    turn_index = Column(Integer, nullable=False)
    strategy = Column(String, nullable=False)
    triggered = Column(Boolean, nullable=False)
    fixation_score = Column(Float, nullable=True)
    avg_similarity = Column(Float, nullable=True)
    dispersion = Column(Float, nullable=True)
    trajectory_movement = Column(Float, nullable=True)
    branches_json = Column(Text, nullable=True)  # JSON exploration tree, if generated
    selected_branch = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ConversationSession", back_populates="events")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

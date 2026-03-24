"""
PostgreSQL Database Models for Short-term Memory.
Stores chat history, attendance, and timetable.
"""
from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean, Date, Time
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class ChatHistory(Base):
    """Chat conversation history."""
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    meta_data = Column(JSONB, nullable=True)  # agent_used, intent, etc. (renamed from metadata)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Attendance(Base):
    """Attendance records."""
    __tablename__ = "attendance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    subject = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)  # present, absent, late
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Timetable(Base):
    """Class/event timetable."""
    __tablename__ = "timetable"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    subject = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    instructor = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    meta_data = Column(JSONB, nullable=True)  # custom fields (renamed from metadata)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

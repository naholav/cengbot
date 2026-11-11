"""
Database Models for CengBot - AI Assistant
==========================================

This module defines the database models and schemas for the CengBot system,
including raw data collection, training data management, and similarity detection.

Features:
- SQLAlchemy ORM models for data persistence
- Cosine similarity checking for duplicate detection
- Automatic answer similarity detection (85% threshold)
- Enhanced database schema with analytics support
- English-only database structure

Author: naholav
"""

import asyncio
import math
import os
import numpy as np
import unicodedata
from collections import Counter
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import (
    create_engine, Column, Integer, BigInteger, String, Text, Boolean, DateTime,
    Float, ForeignKey, text, Index, update, select, case, or_,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime, timezone
import logging
import re
from typing import Dict, Generator, List, Optional, Tuple

# NOTE: this is a library module - it must not configure the root logger
# (a ``logging.basicConfig`` here used to win over the bot's / worker's own
# logging configuration because it ran first at import time).
logger = logging.getLogger(__name__)

# Database connection configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'university_bot.db')}"


def _read_env_database_url() -> Optional[str]:
    """
    Return ``DATABASE_URL`` from ``os.environ``; if it is not set there, fall
    back to the project's ``.env`` file (read-only, ``os.environ`` is not
    modified). The fallback makes every process resolve the same database no
    matter whether ``config.env_loader`` was imported before this module.
    """
    value = os.environ.get("DATABASE_URL")
    if value is not None and value.strip():
        return value.strip()
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.isfile(env_path):
        try:
            from dotenv import dotenv_values
            value = dotenv_values(env_path).get("DATABASE_URL")
        except Exception:  # dotenv missing / unreadable file -> use default
            value = None
        if value is not None and value.strip():
            return value.strip()
    return None


def resolve_database_url(url: Optional[str] = None) -> str:
    """
    Normalize a database URL.

    - empty / ``None`` -> ``<project root>/university_bot.db``
    - relative SQLite paths (``sqlite:///university_bot.db``) are resolved
      against the project root (not the current working directory, which
      differs between the bot, the worker, the API and cron scripts)
    - absolute SQLite paths, in-memory SQLite and non-SQLite URLs are kept
    """
    if url is None or not str(url).strip():
        return DEFAULT_DATABASE_URL
    url = str(url).strip()
    try:
        parsed = make_url(url)
    except Exception:
        logger.error("Invalid DATABASE_URL %r, falling back to the default database", url)
        return DEFAULT_DATABASE_URL
    if parsed.get_backend_name() != "sqlite":
        return url
    database = parsed.database
    if not database or database == ":memory:" or database.startswith("file:"):
        return url
    if os.path.isabs(database):
        return url
    absolute = os.path.normpath(os.path.join(BASE_DIR, database))
    return parsed.set(database=absolute).render_as_string(hide_password=False)


def _create_engine_for(url: str):
    """Create the SQLAlchemy engine (SQLite-only connect args only for SQLite)."""
    kwargs = {"echo": False}
    if make_url(url).get_backend_name() == "sqlite":
        # Sessions are used from the bot's asyncio loop and from pika/worker
        # threads, so a pooled connection may be used by another thread.
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


DATABASE_URL = resolve_database_url(_read_env_database_url())
engine = _create_engine_for(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _utcnow() -> datetime:
    """Naive UTC timestamp (the existing data is stored as naive UTC)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Legacy module-level TF-IDF vectorizer. It is NOT used by the similarity code
# any more (a shared, re-fitted sklearn vectorizer is not thread-safe and its
# English stop-word list is wrong for mostly-Turkish text). Kept only so that
# ``from database_models import vectorizer`` keeps working.
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 2)
)

class RawData(Base):
    """
    Raw data model for storing all user interactions with the bot.
    
    This table stores every question and answer pair from user interactions,
    including metadata for analysis and duplicate detection.
    """
    __tablename__ = "raw_data"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(BigInteger, nullable=False, index=True)
    telegram_message_id = Column(BigInteger)
    username = Column(String(100))
    
    # Core content
    question = Column(Text, nullable=False)
    answer = Column(Text)
    language = Column(String(10))  # 'TR' or 'EN'
    
    # User feedback and approval
    like = Column(Integer)  # -1 (dislike), 1 (like), NULL (no feedback)
    admin_approved = Column(Integer, default=0)  # 0 (pending), 1 (approved)
    
    # Duplicate detection
    is_duplicate = Column(Boolean, default=False)
    duplicate_of_id = Column(Integer, ForeignKey('raw_data.id'))
    similarity_score = Column(Float)  # Cosine similarity score
    
    # Timestamps
    created_at = Column(DateTime, default=_utcnow)
    answered_at = Column(DateTime)
    
    # Additional metadata
    message_thread_id = Column(Integer)  # Telegram topic/thread ID
    processing_time = Column(Float)  # Response time in seconds
    model_version = Column(String(50))  # Model version used
    context_length = Column(Integer)  # Input token count
    response_length = Column(Integer)  # Output token count
    
    # Quality metrics
    quality_score = Column(Float)  # AI quality assessment
    sentiment_score = Column(Float)  # Sentiment analysis
    complexity_score = Column(Integer)  # Question complexity (1-10)
    
    # Categorization
    topic_category = Column(String(100))  # Automated topic classification
    keywords = Column(Text)  # Extracted keywords (JSON format)
    
    # Admin features
    admin_notes = Column(Text)  # Admin comments
    last_updated = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    
    # Relationships
    training_data = relationship("TrainingData", back_populates="source")

class TrainingData(Base):
    """
    Training data model for storing approved question-answer pairs.
    
    This table contains curated data that has been approved by administrators
    and is used for model training and fine-tuning.
    """
    __tablename__ = "training_data"
    
    # Primary identification
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey('raw_data.id'), unique=True)
    
    # Core content
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    language = Column(String(10))  # 'TR' or 'EN'
    
    # Duplicate detection for answers
    is_answer_duplicate = Column(Boolean, default=False)
    duplicate_answer_of_id = Column(Integer, ForeignKey('training_data.id'))
    answer_similarity_score = Column(Float)  # Cosine similarity for answers
    
    # Quality and usage metrics
    quality_score = Column(Integer)  # 1-10 quality rating
    usage_count = Column(Integer, default=0)  # Training usage frequency
    effectiveness_score = Column(Float)  # Training effectiveness
    
    # Categorization
    category = Column(String(100))  # Question category
    difficulty_level = Column(Integer)  # Complexity level (1-10)
    topic_tags = Column(Text)  # Topic tags (JSON format)
    
    # Training metadata
    training_weight = Column(Float, default=1.0)  # Training sample weight
    augmentation_count = Column(Integer, default=0)  # Paraphrase variations
    source_type = Column(String(50))  # 'original', 'augmented', 'manual'
    
    # Review and approval
    review_status = Column(String(20), default='pending')  # 'pending', 'approved', 'rejected'
    reviewer_id = Column(String(100))  # Admin who reviewed
    review_date = Column(DateTime)  # Review timestamp
    admin_notes = Column(Text)  # Admin comments
    
    # Versioning and status
    version = Column(Integer, default=1)  # Version tracking
    is_active = Column(Boolean, default=True)  # Active in training
    
    # Timestamps
    created_at = Column(DateTime, default=_utcnow)
    last_used_at = Column(DateTime)  # Last training run
    last_updated = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    
    # Relationships
    source = relationship("RawData", back_populates="training_data")

# Database indexes for performance optimization
Index('idx_raw_data_telegram_id', RawData.telegram_id)
Index('idx_raw_data_created_at', RawData.created_at)
Index('idx_raw_data_admin_approved', RawData.admin_approved)
Index('idx_raw_data_like', RawData.like)
Index('idx_raw_data_language', RawData.language)
Index('idx_raw_data_is_duplicate', RawData.is_duplicate)
Index('idx_raw_data_composite', RawData.telegram_id, RawData.created_at)

Index('idx_training_data_created_at', TrainingData.created_at)
Index('idx_training_data_source_id', TrainingData.source_id)
Index('idx_training_data_language', TrainingData.language)
Index('idx_training_data_review_status', TrainingData.review_status)
Index('idx_training_data_is_active', TrainingData.is_active)

class UserVotes(Base):
    """
    User votes model for tracking user votes on questions with change limits.
    """
    __tablename__ = "user_votes"
    
    id = Column(Integer, primary_key=True, index=True)
    raw_data_id = Column(Integer, ForeignKey('raw_data.id'), nullable=False)
    telegram_user_id = Column(BigInteger, nullable=False, index=True)
    current_vote = Column(Integer)  # 1 (like), -1 (dislike), NULL (no vote)
    vote_changes_count = Column(Integer, default=0)  # Max 2 changes allowed
    
    # Timestamps
    first_vote_at = Column(DateTime, default=_utcnow)
    last_vote_at = Column(DateTime, default=_utcnow)
    created_at = Column(DateTime, default=_utcnow)
    
    # One vote row per (question, user). The UNIQUE index is what makes
    # concurrent double clicks safe; it has the same name as the one init_db()
    # has always created on existing databases, so nothing changes there.
    __table_args__ = (
        Index('idx_user_votes_composite', 'raw_data_id', 'telegram_user_id'),
        Index('idx_user_votes_unique', 'raw_data_id', 'telegram_user_id', unique=True),
    )

class UserAnalytics(Base):
    """
    User analytics model for tracking user behavior and engagement.
    """
    __tablename__ = "user_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(BigInteger, nullable=False, unique=True, index=True)
    
    # Usage statistics
    session_count = Column(Integer, default=0)
    total_questions = Column(Integer, default=0)
    total_likes = Column(Integer, default=0)
    total_dislikes = Column(Integer, default=0)
    
    # Preferences
    preferred_language = Column(String(10))
    most_active_hour = Column(Integer)
    avg_session_duration = Column(Float)
    
    # Engagement metrics
    avg_satisfaction = Column(Float)
    engagement_score = Column(Float)
    retention_score = Column(Float)
    
    # Timestamps
    first_interaction = Column(DateTime)
    last_interaction = Column(DateTime)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

class SystemMetrics(Base):
    """
    System metrics model for monitoring performance and health.
    """
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_type = Column(String(50))  # 'counter', 'gauge', 'histogram'
    
    # Metadata
    service_name = Column(String(50))  # Service that generated the metric
    instance_id = Column(String(50))  # Instance identifier
    tags = Column(Text)  # Additional tags (JSON format)
    
    # Timestamps
    timestamp = Column(DateTime, default=_utcnow)
    
    # Additional details
    details = Column(Text)  # Additional metric details (JSON format)

def clean_text(text: str) -> str:
    """
    Clean text for similarity comparison.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Convert to lowercase ('İ'.lower() would give 'i' + U+0307 combining dot)
    text = text.replace('İ', 'i').lower()

    return text


# --- Similarity internals ------------------------------------------------------
#
# The score is the cosine similarity of the TF-IDF vectors of the two texts,
# computed exactly like the previous implementation (a TfidfVectorizer fitted
# on ``[text1, text2, "dummy document text"]``: raw term counts, smooth idf
# ``ln((1 + n) / (1 + df)) + 1`` with n = 3, L2-normalised rows), so the
# thresholds (0.45 for questions, 0.85 for answers) and the scores already
# stored in the database keep their meaning. It is computed in pure Python
# instead of fitting a new sklearn vectorizer for every pair (~50x faster, no
# shared mutable state -> thread-safe, never raises "empty vocabulary").
#
# Tokenisation fixes:
# - Unicode letters AND digits are tokens. Digits used to be dropped, so
#   "Fizik 3 kaç akts" vs "Fizik 4 kaç akts" (or "BIL 341" vs "BIL 342")
#   scored 1.0.
# - Turkish-aware folding: case-insensitive ("İ"/"I"/"ı"/"i" all match) and
#   diacritic-insensitive ("ogrenci" == "öğrenci"), since users frequently type
#   without a Turkish keyboard.

_TOKEN_RE = re.compile(r"[^\W_]+")
_DUMMY_DOC_TOKENS = frozenset(("dummy", "document", "text"))
# idf for a term contained in 1, 2 or 3 of the 3 documents
_IDF_BY_DF = {df: math.log((1 + 3) / (1 + df)) + 1.0 for df in (1, 2, 3)}


def _fold_for_matching(text: str) -> str:
    """Lowercase and strip diacritics (Turkish dotted/dotless i aware)."""
    text = text.replace("İ", "i").replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


@lru_cache(maxsize=8192)
def _prepare(text: Optional[str]) -> Tuple[str, Tuple[str, ...], Tuple[Tuple[str, int], ...]]:
    """
    Pre-process a text once: (cleaned text, tokens, sorted term counts).
    Cached because the same rows are compared again and again (every new
    question is compared with the previous 100, the admin re-check compares
    everything with everything). Returns only immutable values.
    """
    cleaned = clean_text(text) if text else ""
    if not cleaned:
        return "", (), ()
    tokens = tuple(_TOKEN_RE.findall(_fold_for_matching(cleaned)))
    counts = tuple(sorted(Counter(tokens).items()))
    return cleaned, tokens, counts


def _tfidf_cosine(counts1: Dict[str, int], counts2: Dict[str, int]) -> float:
    """TF-IDF cosine of two term-count dicts (see the comment block above)."""
    if not counts1 or not counts2:
        return 0.0
    shared = counts1.keys() & counts2.keys()
    if not shared:
        return 0.0

    def weight(term: str, count: int) -> float:
        df = 1 + (term in counts1 and term in counts2) + (term in _DUMMY_DOC_TOKENS)
        return count * _IDF_BY_DF[df]

    norm1 = math.sqrt(sum(weight(t, c) ** 2 for t, c in counts1.items()))
    norm2 = math.sqrt(sum(weight(t, c) ** 2 for t, c in counts2.items()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    dot = sum(weight(t, counts1[t]) * weight(t, counts2[t]) for t in shared)
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


def _similarity(prepared1, prepared2) -> float:
    """Similarity of two ``_prepare``-d texts, in [0.0, 1.0]."""
    cleaned1, tokens1, counts1 = prepared1
    cleaned2, tokens2, counts2 = prepared2
    if not cleaned1 or not cleaned2:
        return 0.0
    if cleaned1 == cleaned2 or (tokens1 and tokens1 == tokens2):
        return 1.0
    if counts1 and counts2:
        return _tfidf_cosine(dict(counts1), dict(counts2))
    if counts1 or counts2:
        return 0.0
    # Neither text contains a letter or digit (punctuation only):
    # Jaccard similarity of the whitespace-separated chunks.
    words1, words2 = set(cleaned1.split()), set(cleaned2.split())
    union = words1 | words2
    return len(words1 & words2) / len(union) if union else 0.0


def _safe_rollback(db: Session) -> None:
    """Roll back after an error so the caller's session stays usable."""
    try:
        db.rollback()
    except Exception:  # pragma: no cover - rollback itself failed
        logger.exception("Rollback failed")


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using TF-IDF vectors.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    try:
        score = _similarity(_prepare(text1), _prepare(text2))
        logger.debug("Similarity calculated: %.3f", score)
        return score
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def find_similar_questions(db: Session, question: str, threshold: float = 0.45, limit: int = 100,
                           exclude_id: Optional[int] = None,
                           before_id: Optional[int] = None) -> List[Tuple[int, str, float]]:
    """
    Find similar questions in the raw_data table using cosine similarity.
    Only a bounded window of rows is compared (``limit``), so the cost per new
    question does not grow with the table.

    Args:
        db: Database session
        question: Question text to compare
        threshold: Similarity threshold (default: 0.45)
        limit: Maximum number of questions to compare (default: 100;
            None or <= 0 compares all)
        exclude_id: Optional raw_data id to leave out (the question itself)
        before_id: If given, only questions with ``id < before_id`` are
            compared (the window is the ``limit`` questions right before it)

    Returns:
        List of tuples (id, question, similarity_score), most similar first
    """
    try:
        target = _prepare(question)
        if not target[0]:
            return []

        query = db.query(RawData.id, RawData.question)
        if exclude_id is not None:
            query = query.filter(RawData.id != exclude_id)
        if before_id is not None:
            query = query.filter(RawData.id < before_id).order_by(RawData.id.desc())
        else:
            # Recent questions first (more likely to be similar)
            query = query.order_by(RawData.created_at.desc(), RawData.id.desc())
        if limit is not None and limit > 0:
            query = query.limit(limit)

        similar_questions = []
        for existing_id, existing_question in query.all():
            # Skip empty questions
            if not existing_question or not existing_question.strip():
                continue
            similarity = _similarity(target, _prepare(existing_question))
            if similarity >= threshold:
                similar_questions.append((existing_id, existing_question, similarity))

        # Most similar first; ties -> oldest first (deterministic)
        similar_questions.sort(key=lambda x: (-x[2], x[0]))
        return similar_questions

    except Exception as e:
        logger.error(f"Error finding similar questions: {e}")
        _safe_rollback(db)
        return []


def find_similar_answers(db: Session, answer: str, threshold: float = 0.85,
                         limit: Optional[int] = None,
                         exclude_id: Optional[int] = None,
                         before_id: Optional[int] = None) -> List[Tuple[int, str, float]]:
    """
    Find similar answers in the training_data table.

    Args:
        db: Database session
        answer: Answer text to compare
        threshold: Similarity threshold (default: 0.85)
        limit: Optional maximum number of (most recent) entries to compare
        exclude_id: Optional training_data id to leave out
        before_id: If given, only entries with ``id < before_id`` are compared

    Returns:
        List of tuples (id, answer, similarity_score), most similar first
    """
    try:
        target = _prepare(answer)
        if not target[0]:
            return []

        query = db.query(TrainingData.id, TrainingData.answer)
        if exclude_id is not None:
            query = query.filter(TrainingData.id != exclude_id)
        if before_id is not None:
            query = query.filter(TrainingData.id < before_id)
        query = query.order_by(TrainingData.id.desc())
        if limit is not None and limit > 0:
            query = query.limit(limit)

        similar_answers = []
        for existing_id, existing_answer in query.all():
            if not existing_answer or not existing_answer.strip():
                continue
            similarity = _similarity(target, _prepare(existing_answer))
            if similarity >= threshold:
                similar_answers.append((existing_id, existing_answer, similarity))

        # Most similar first; ties -> oldest first (deterministic)
        similar_answers.sort(key=lambda x: (-x[2], x[0]))
        return similar_answers

    except Exception as e:
        logger.error(f"Error finding similar answers: {e}")
        _safe_rollback(db)
        return []


def _resolve_question_root(db: Session, start_id: int, current_id: int) -> int:
    """
    Follow ``duplicate_of_id`` links from ``start_id`` to the original question,
    so duplicate groups stay flat. Guards against cycles / dangling links in
    legacy data (never returns ``current_id``).
    """
    root_id = start_id
    seen = {current_id, start_id}
    for _ in range(100):
        row = db.query(RawData.is_duplicate, RawData.duplicate_of_id)\
            .filter(RawData.id == root_id).first()
        if row is None or not row.is_duplicate or row.duplicate_of_id is None:
            break
        next_id = row.duplicate_of_id
        if next_id in seen or db.query(RawData.id).filter(RawData.id == next_id).first() is None:
            break
        seen.add(next_id)
        root_id = next_id
    return root_id


def mark_duplicate_questions(db: Session, question_id: int, question_text: str) -> bool:
    """
    Check for duplicate questions and mark them accordingly.
    Uses oldest question as the original reference.

    Only questions asked *before* ``question_id`` can be its original (the
    window is the 100 questions right before it), so duplicate links always
    point to an older question: no cycles, and the first question of a group
    is never flagged itself - also when the admin re-runs detection over the
    whole table. Only the given question's row is modified.

    Args:
        db: Database session
        question_id: ID of the question to check
        question_text: Text of the question

    Returns:
        True if duplicates were found and marked
    """
    try:
        similar_questions = find_similar_questions(db, question_text, threshold=0.45,
                                                   before_id=question_id)
        if not similar_questions:
            return False

        current_question = db.query(RawData).filter(RawData.id == question_id).first()
        if not current_question:
            return False

        # The oldest matching question (smallest id) is the original reference
        match_id, _, similarity_score = min(similar_questions, key=lambda sq: sq[0])
        original_id = _resolve_question_root(db, match_id, question_id)

        current_question.is_duplicate = True
        current_question.duplicate_of_id = original_id
        current_question.similarity_score = similarity_score
        db.commit()

        logger.info(f"Question {question_id} marked as duplicate of {original_id} (similarity: {similarity_score:.3f})")
        return True

    except Exception as e:
        logger.error(f"Error marking duplicate questions: {e}")
        _safe_rollback(db)
        return False


def mark_duplicate_answers(db: Session, training_id: int, answer_text: str) -> bool:
    """
    Check for duplicate answers in training data and mark them accordingly.

    The entry is marked as a duplicate of the most similar *older* entry
    (``id < training_id``), so two entries can never be marked as duplicates
    of each other. Only the given entry's row is modified.

    Args:
        db: Database session
        training_id: ID of the training data entry
        answer_text: Text of the answer

    Returns:
        True if duplicates were found and marked
    """
    try:
        similar_answers = find_similar_answers(db, answer_text, threshold=0.85,
                                               before_id=training_id)
        if not similar_answers:
            return False

        current_training = db.query(TrainingData).filter(TrainingData.id == training_id).first()
        if not current_training:
            return False

        # Mark as duplicate of the most similar answer
        original_id, _, similarity_score = similar_answers[0]

        current_training.is_answer_duplicate = True
        current_training.duplicate_answer_of_id = original_id
        current_training.answer_similarity_score = similarity_score
        db.commit()

        logger.info(f"Answer {training_id} marked as duplicate of {original_id} (similarity: {similarity_score:.3f})")
        return True

    except Exception as e:
        logger.error(f"Error marking duplicate answers: {e}")
        _safe_rollback(db)
        return False


def process_new_question(db: Session, question_data: dict) -> RawData:
    """
    Process a new question with duplicate detection.

    Args:
        db: Database session
        question_data: Dictionary containing question data

    Returns:
        Created RawData instance
    """
    try:
        # Create new raw data entry
        raw_data = RawData(**question_data)
        db.add(raw_data)
        db.commit()
        db.refresh(raw_data)
    except Exception as e:
        logger.error(f"Error processing new question: {e}")
        db.rollback()
        raise

    # Check for duplicates (best effort: never loses the saved question)
    mark_duplicate_questions(db, raw_data.id, raw_data.question)
    return raw_data


def process_new_training_data(db: Session, training_data: dict) -> TrainingData:
    """
    Process new training data with duplicate answer detection.

    Args:
        db: Database session
        training_data: Dictionary containing training data

    Returns:
        Created TrainingData instance
    """
    try:
        # Create new training data entry
        training_entry = TrainingData(**training_data)
        db.add(training_entry)
        db.commit()
        db.refresh(training_entry)
    except Exception as e:
        logger.error(f"Error processing new training data: {e}")
        db.rollback()
        raise

    # Check for duplicate answers (best effort)
    mark_duplicate_answers(db, training_entry.id, training_entry.answer)
    return training_entry


MAX_VOTE_CHANGES = 2
_VALID_VOTES = (1, -1)


def _like_value_expression(raw_data_id_expr):
    """SQL expression: 1 / -1 / NULL from the sign of (likes - dislikes)."""
    score = select(func.coalesce(func.sum(UserVotes.current_vote), 0))\
        .where(UserVotes.raw_data_id == raw_data_id_expr,
               UserVotes.current_vote.in_(_VALID_VOTES))\
        .scalar_subquery()
    return case((score > 0, 1), (score < 0, -1), else_=None)


def _sync_raw_data_like(db: Session, raw_data_id: int) -> None:
    """
    Keep ``raw_data.like`` (shown by the admin panel / statistics) in sync with
    the per-user votes: 1 if likes > dislikes, -1 if dislikes > likes, NULL
    otherwise. Single UPDATE statement, runs in the caller's transaction.
    """
    db.execute(
        update(RawData)
        .where(RawData.id == raw_data_id)
        .values(like=_like_value_expression(raw_data_id))
        .execution_options(synchronize_session=False)
    )


def sync_like_from_votes(db: Session, raw_data_id: Optional[int] = None) -> int:
    """
    Recompute ``raw_data.like`` from ``user_votes`` and commit.

    Args:
        db: Database session
        raw_data_id: Only this question; None = every question that has votes
            (one-off backfill for rows voted on before the sync existed)

    Returns:
        Number of raw_data rows updated
    """
    try:
        if raw_data_id is not None:
            stmt = update(RawData).where(RawData.id == raw_data_id)\
                .values(like=_like_value_expression(raw_data_id))
        else:
            voted_ids = select(UserVotes.raw_data_id).distinct()
            stmt = update(RawData).where(RawData.id.in_(voted_ids))\
                .values(like=_like_value_expression(RawData.id))
        result = db.execute(stmt.execution_options(synchronize_session=False))
        db.commit()
        db.expire_all()
        return result.rowcount or 0
    except Exception:
        _safe_rollback(db)
        raise


def _vote_text(vote_type: int) -> str:
    return "👍 Liked!" if vote_type == 1 else "👎 Disliked!"


def handle_user_vote(db: Session, raw_data_id: int, telegram_user_id: int, vote_type: int) -> dict:
    """
    Handle user vote with 2 change limit system.

    Race-safe: a concurrent double click cannot create two vote rows (UNIQUE
    index + retry) nor use the same vote change twice (conditional UPDATE).
    ``raw_data.like`` is updated in the same transaction.

    Args:
        db: Database session
        raw_data_id: ID of the raw data question
        telegram_user_id: Telegram user ID
        vote_type: 1 (like), -1 (dislike)

    Returns:
        Dict with success status and message
    """
    if vote_type not in _VALID_VOTES:
        return {"success": False, "message": "Invalid vote", "changes_left": 0}

    try:
        if db.query(RawData.id).filter(RawData.id == raw_data_id).first() is None:
            return {"success": False, "message": "Record not found!", "changes_left": 0}

        for _attempt in range(3):
            # Check if user has already voted
            existing_vote = db.query(UserVotes).filter(
                UserVotes.raw_data_id == raw_data_id,
                UserVotes.telegram_user_id == telegram_user_id
            ).order_by(UserVotes.id).populate_existing().first()

            if existing_vote is None:
                # Create new vote
                db.add(UserVotes(
                    raw_data_id=raw_data_id,
                    telegram_user_id=telegram_user_id,
                    current_vote=vote_type,
                    vote_changes_count=0
                ))
                try:
                    db.flush()
                except IntegrityError:
                    # A concurrent request inserted the vote first -> re-read
                    db.rollback()
                    continue
                _sync_raw_data_like(db, raw_data_id)
                db.commit()
                return {
                    "success": True,
                    "message": f"{_vote_text(vote_type)} ({MAX_VOTE_CHANGES} changes available)",
                    "changes_left": MAX_VOTE_CHANGES
                }

            used_changes = existing_vote.vote_changes_count or 0

            # Check if trying to vote the same thing
            if existing_vote.current_vote == vote_type:
                return {
                    "success": False,
                    "message": "You already voted this way!",
                    "changes_left": max(0, MAX_VOTE_CHANGES - used_changes)
                }

            # Check if user has exceeded change limit
            if used_changes >= MAX_VOTE_CHANGES:
                return {
                    "success": False,
                    "message": f"No vote changes left! ({MAX_VOTE_CHANGES}/{MAX_VOTE_CHANGES} used)",
                    "changes_left": 0
                }

            # Conditional update: only succeeds if nobody changed the row since
            # we read it (protects the change limit against double clicks).
            result = db.execute(
                update(UserVotes)
                .where(
                    UserVotes.id == existing_vote.id,
                    func.coalesce(UserVotes.vote_changes_count, 0) == used_changes,
                    or_(UserVotes.current_vote.is_(None), UserVotes.current_vote != vote_type),
                )
                .values(
                    current_vote=vote_type,
                    vote_changes_count=used_changes + 1,
                    last_vote_at=_utcnow(),
                )
                .execution_options(synchronize_session=False)
            )
            if result.rowcount != 1:
                db.rollback()
                continue

            _sync_raw_data_like(db, raw_data_id)
            db.commit()

            changes_left = MAX_VOTE_CHANGES - (used_changes + 1)
            vote_text = _vote_text(vote_type)
            if changes_left == 0:
                message = f"{vote_text} (Last change!)"
            else:
                message = f"{vote_text} ({changes_left} changes left)"
            return {
                "success": True,
                "message": message,
                "changes_left": changes_left
            }

        logger.warning(f"Vote for raw_data {raw_data_id} by {telegram_user_id} kept conflicting, giving up")
        return {
            "success": False,
            "message": "Error saving vote",
            "changes_left": 0
        }

    except Exception as e:
        logger.error(f"Error handling user vote: {e}")
        _safe_rollback(db)
        return {
            "success": False,
            "message": "Error saving vote",
            "changes_left": 0
        }


def get_vote_statistics(db: Session, raw_data_id: int) -> dict:
    """
    Get vote statistics for a specific question.

    Args:
        db: Database session
        raw_data_id: ID of the raw data question

    Returns:
        Dict with vote statistics
    """
    try:
        total, likes, dislikes = db.query(
            func.count(UserVotes.id),
            func.coalesce(func.sum(case((UserVotes.current_vote == 1, 1), else_=0)), 0),
            func.coalesce(func.sum(case((UserVotes.current_vote == -1, 1), else_=0)), 0),
        ).filter(UserVotes.raw_data_id == raw_data_id).one()
        likes, dislikes = int(likes), int(dislikes)
        return {
            "likes": likes,
            "dislikes": dislikes,
            "total_votes": int(total),
            "score": likes - dislikes
        }
    except Exception as e:
        logger.error(f"Error getting vote statistics: {e}")
        _safe_rollback(db)
        return {
            "likes": 0,
            "dislikes": 0,
            "total_votes": 0,
            "score": 0
        }


def get_database_statistics(db: Session) -> dict:
    """
    Get database statistics.

    Args:
        db: Database session

    Returns:
        Dictionary containing database statistics
    """
    try:
        stats = {}

        # Raw data statistics
        stats['total_questions'] = db.query(RawData).count()
        stats['answered_questions'] = db.query(RawData).filter(RawData.answer.isnot(None)).count()
        stats['liked_questions'] = db.query(RawData).filter(RawData.like == 1).count()
        stats['disliked_questions'] = db.query(RawData).filter(RawData.like == -1).count()
        stats['approved_questions'] = db.query(RawData).filter(RawData.admin_approved == 1).count()
        stats['duplicate_questions'] = db.query(RawData).filter(RawData.is_duplicate == True).count()

        # Training data statistics
        stats['training_data_count'] = db.query(TrainingData).count()
        stats['active_training_data'] = db.query(TrainingData).filter(TrainingData.is_active == True).count()
        stats['duplicate_answers'] = db.query(TrainingData).filter(TrainingData.is_answer_duplicate == True).count()

        # Language distribution
        language_stats = db.query(RawData.language, func.count(RawData.id)).group_by(RawData.language).all()
        stats['language_distribution'] = {lang: count for lang, count in language_stats}

        # Quality metrics
        avg_quality = db.query(func.avg(RawData.quality_score)).scalar()
        stats['avg_quality_score'] = float(avg_quality) if avg_quality else 0.0

        # Response time metrics
        avg_response_time = db.query(func.avg(RawData.processing_time)).scalar()
        stats['avg_response_time'] = float(avg_response_time) if avg_response_time else 0.0

        return stats

    except Exception as e:
        logger.error(f"Error getting database statistics: {e}")
        _safe_rollback(db)
        return {}


# Indexes that existing databases may predate (CREATE TABLE IF NOT EXISTS via
# create_all() does not add indexes to tables that already exist).
_INIT_INDEX_STATEMENTS = (
    # Raw data indexes
    "CREATE INDEX IF NOT EXISTS idx_raw_data_telegram_id ON raw_data(telegram_id);",
    "CREATE INDEX IF NOT EXISTS idx_raw_data_language ON raw_data(language);",
    "CREATE INDEX IF NOT EXISTS idx_raw_data_created_at ON raw_data(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_raw_data_is_duplicate ON raw_data(is_duplicate);",
    "CREATE INDEX IF NOT EXISTS idx_raw_data_admin_approved ON raw_data(admin_approved);",
    # Training data indexes
    "CREATE INDEX IF NOT EXISTS idx_training_data_source_id ON training_data(source_id);",
    "CREATE INDEX IF NOT EXISTS idx_training_data_language ON training_data(language);",
    "CREATE INDEX IF NOT EXISTS idx_training_data_is_active ON training_data(is_active);",
    "CREATE INDEX IF NOT EXISTS idx_training_data_is_answer_duplicate ON training_data(is_answer_duplicate);",
    "CREATE INDEX IF NOT EXISTS idx_training_data_review_status ON training_data(review_status);",
    # User analytics indexes
    "CREATE INDEX IF NOT EXISTS idx_user_analytics_telegram_id ON user_analytics(telegram_id);",
    # User votes indexes
    "CREATE INDEX IF NOT EXISTS idx_user_votes_raw_data_id ON user_votes(raw_data_id);",
    "CREATE INDEX IF NOT EXISTS idx_user_votes_telegram_user_id ON user_votes(telegram_user_id);",
    # System metrics indexes
    "CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);",
    "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);",
)
_UNIQUE_VOTE_INDEX_STATEMENT = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_votes_unique "
    "ON user_votes(raw_data_id, telegram_user_id);"
)


def init_db():
    """
    Initialize database and create all tables.
    """
    try:
        # Create all tables (uses the module-level engine at call time)
        Base.metadata.create_all(bind=engine)

        # Create indexes for performance
        with engine.begin() as conn:
            for statement in _INIT_INDEX_STATEMENTS:
                conn.execute(text(statement))

        # One vote per user and question. If an old database already holds
        # duplicate vote rows this fails; keep the system running and say why.
        try:
            with engine.begin() as conn:
                conn.execute(text(_UNIQUE_VOTE_INDEX_STATEMENT))
        except IntegrityError as e:
            logger.error(
                "Could not create UNIQUE index idx_user_votes_unique: user_votes "
                f"contains duplicate (raw_data_id, telegram_user_id) rows. "
                f"Remove the duplicates and restart. ({e})"
            )

        logger.info("SQLite database initialized successfully with all tables and indexes!")
        print("✅ Database initialized successfully!")

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.

    Returns:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Export all important functions and classes
__all__ = [
    'RawData',
    'TrainingData', 
    'UserVotes',
    'UserAnalytics',
    'SystemMetrics',
    'init_db',
    'get_db',
    'process_new_question',
    'process_new_training_data',
    'mark_duplicate_questions',
    'mark_duplicate_answers',
    'find_similar_questions',
    'find_similar_answers',
    'calculate_cosine_similarity',
    'handle_user_vote',
    'get_vote_statistics',
    'get_database_statistics',
    'sync_like_from_votes',
]
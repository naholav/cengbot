"""
Database Models for CengBot - AI Assistant
==========================================

This module defines the database models and schemas for the CengBot system,
including raw data collection, training data management, and similarity detection.

Features:
- SQLAlchemy ORM models for data persistence
- Cosine similarity checking for duplicate detection
- Automatic answer similarity detection (80% threshold)
- Enhanced database schema with analytics support
- English-only database structure

Author: naholav
"""

import asyncio
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Text, Boolean, DateTime, Float, ForeignKey, text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from datetime import datetime
import logging
import re
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection configuration
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'university_bot.db')}"
engine = create_engine(
    DATABASE_URL, 
    echo=False, 
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global TF-IDF vectorizer for similarity calculations
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
    created_at = Column(DateTime, default=datetime.utcnow)
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
    last_updated = Column(DateTime, default=datetime.utcnow)
    
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
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime)  # Last training run
    last_updated = Column(DateTime, default=datetime.utcnow)
    
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
    first_vote_at = Column(DateTime, default=datetime.utcnow)
    last_vote_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite unique constraint
    __table_args__ = (
        Index('idx_user_votes_composite', 'raw_data_id', 'telegram_user_id'),
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

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
    timestamp = Column(DateTime, default=datetime.utcnow)
    
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
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Convert to lowercase
    text = text.lower()
    
    return text

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
        # Clean texts
        clean_text1 = clean_text(text1)
        clean_text2 = clean_text(text2)
        
        # Handle empty texts
        if not clean_text1 or not clean_text2:
            return 0.0
        
        # If texts are identical, return 1.0
        if clean_text1 == clean_text2:
            return 1.0
        
        # Simple word-based similarity using set intersection
        words1 = set(clean_text1.split())
        words2 = set(clean_text2.split())
        
        # Calculate Jaccard similarity as a fallback
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
            
        jaccard_score = len(intersection) / len(union)
        
        # Try TF-IDF as primary method
        try:
            # Add a dummy document to ensure min_df works
            dummy_doc = "dummy document text"
            texts = [clean_text1, clean_text2, dummy_doc]
            
            # Create vectorizer
            local_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,
                lowercase=True,
                ngram_range=(1, 1),  # Only unigrams for simplicity
                min_df=1,
                token_pattern=r'[a-zA-ZçÇğĞıIİöÖşŞüÜ]+',
                analyzer='word'
            )
            
            # Calculate TF-IDF vectors
            tfidf_matrix = local_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity between first two documents
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            tfidf_score = float(similarity_matrix[0][0])
            
            # Use TF-IDF score if it's reasonable, otherwise use Jaccard
            final_score = tfidf_score if tfidf_score > 0.0 else jaccard_score
            
        except Exception as tfidf_error:
            logger.warning(f"TF-IDF failed, using Jaccard similarity: {tfidf_error}")
            final_score = jaccard_score
        
        logger.info(f"Similarity calculated: {final_score:.3f} (Jaccard: {jaccard_score:.3f}) for texts: '{clean_text1[:50]}...' vs '{clean_text2[:50]}...'")
        return final_score
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def find_similar_questions(db: Session, question: str, threshold: float = 0.45, limit: int = 100) -> List[Tuple[int, str, float]]:
    """
    Find similar questions in the raw_data table using cosine similarity.
    Optimized to limit database queries and improve performance.
    
    Args:
        db: Database session
        question: Question text to compare
        threshold: Similarity threshold (default: 0.8)
        limit: Maximum number of questions to compare (default: 100)
        
    Returns:
        List of tuples (id, question, similarity_score)
    """
    try:
        # Get recent questions first (more likely to be similar)
        existing_questions = db.query(RawData.id, RawData.question)\
            .order_by(RawData.created_at.desc())\
            .limit(limit)\
            .all()
        
        similar_questions = []
        
        # Pre-process the input question for better comparison
        question_cleaned = clean_text(question)
        
        for existing_id, existing_question in existing_questions:
            # Skip empty questions
            if not existing_question or not existing_question.strip():
                continue
                
            # Calculate similarity
            similarity = calculate_cosine_similarity(question_cleaned, clean_text(existing_question))
            
            # Check if similarity exceeds threshold
            if similarity >= threshold:
                similar_questions.append((existing_id, existing_question, similarity))
        
        # Sort by similarity score (descending)
        similar_questions.sort(key=lambda x: x[2], reverse=True)
        
        return similar_questions
        
    except Exception as e:
        logger.error(f"Error finding similar questions: {e}")
        return []

def find_similar_answers(db: Session, answer: str, threshold: float = 0.85) -> List[Tuple[int, str, float]]:
    """
    Find similar answers in the training_data table.
    
    Args:
        db: Database session
        answer: Answer text to compare
        threshold: Similarity threshold (default: 0.8)
        
    Returns:
        List of tuples (id, answer, similarity_score)
    """
    try:
        # Get all answers from training data
        existing_answers = db.query(TrainingData.id, TrainingData.answer).all()
        
        similar_answers = []
        
        for existing_id, existing_answer in existing_answers:
            # Calculate similarity
            similarity = calculate_cosine_similarity(answer, existing_answer)
            
            # Check if similarity exceeds threshold
            if similarity >= threshold:
                similar_answers.append((existing_id, existing_answer, similarity))
        
        # Sort by similarity score (descending)
        similar_answers.sort(key=lambda x: x[2], reverse=True)
        
        return similar_answers
        
    except Exception as e:
        logger.error(f"Error finding similar answers: {e}")
        return []

def mark_duplicate_questions(db: Session, question_id: int, question_text: str) -> bool:
    """
    Check for duplicate questions and mark them accordingly.
    Uses oldest question as the original reference.
    
    Args:
        db: Database session
        question_id: ID of the question to check
        question_text: Text of the question
        
    Returns:
        True if duplicates were found and marked
    """
    try:
        # Find similar questions
        similar_questions = find_similar_questions(db, question_text, threshold=0.45)
        
        # Remove the current question from results
        similar_questions = [sq for sq in similar_questions if sq[0] != question_id]
        
        if similar_questions:
            # Get the current question
            current_question = db.query(RawData).filter(RawData.id == question_id).first()
            
            if current_question:
                # Find the oldest question (smallest ID) as the original
                oldest_id = min([sq[0] for sq in similar_questions])
                similarity_score = next(sq[2] for sq in similar_questions if sq[0] == oldest_id)
                
                # Update all similar questions to reference the oldest one
                for similar_id, _, _ in similar_questions:
                    if similar_id != oldest_id:
                        similar_question = db.query(RawData).filter(RawData.id == similar_id).first()
                        if similar_question:
                            similar_question.is_duplicate = True
                            similar_question.duplicate_of_id = oldest_id
                            similar_question.similarity_score = similarity_score
                
                # Mark current question as duplicate of the oldest
                current_question.is_duplicate = True
                current_question.duplicate_of_id = oldest_id
                current_question.similarity_score = similarity_score
                
                db.commit()
                
                logger.info(f"Question {question_id} marked as duplicate of {oldest_id} (similarity: {similarity_score:.3f})")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error marking duplicate questions: {e}")
        return False

def mark_duplicate_answers(db: Session, training_id: int, answer_text: str) -> bool:
    """
    Check for duplicate answers in training data and mark them accordingly.
    
    Args:
        db: Database session
        training_id: ID of the training data entry
        answer_text: Text of the answer
        
    Returns:
        True if duplicates were found and marked
    """
    try:
        # Find similar answers
        similar_answers = find_similar_answers(db, answer_text, threshold=0.85)
        
        # Remove the current answer from results
        similar_answers = [sa for sa in similar_answers if sa[0] != training_id]
        
        if similar_answers:
            # Get the current training data
            current_training = db.query(TrainingData).filter(TrainingData.id == training_id).first()
            
            if current_training:
                # Mark as duplicate of the most similar answer
                original_id, _, similarity_score = similar_answers[0]
                
                current_training.is_answer_duplicate = True
                current_training.duplicate_answer_of_id = original_id
                current_training.answer_similarity_score = similarity_score
                
                db.commit()
                
                logger.info(f"Answer {training_id} marked as duplicate of {original_id} (similarity: {similarity_score:.3f})")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error marking duplicate answers: {e}")
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
        
        # Check for duplicates
        mark_duplicate_questions(db, raw_data.id, raw_data.question)
        
        return raw_data
        
    except Exception as e:
        logger.error(f"Error processing new question: {e}")
        db.rollback()
        raise

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
        
        # Check for duplicate answers
        mark_duplicate_answers(db, training_entry.id, training_entry.answer)
        
        return training_entry
        
    except Exception as e:
        logger.error(f"Error processing new training data: {e}")
        db.rollback()
        raise

def handle_user_vote(db: Session, raw_data_id: int, telegram_user_id: int, vote_type: int) -> dict:
    """
    Handle user vote with 2 change limit system.
    
    Args:
        db: Database session
        raw_data_id: ID of the raw data question
        telegram_user_id: Telegram user ID
        vote_type: 1 (like), -1 (dislike)
        
    Returns:
        Dict with success status and message
    """
    try:
        # Check if user has already voted
        existing_vote = db.query(UserVotes).filter(
            UserVotes.raw_data_id == raw_data_id,
            UserVotes.telegram_user_id == telegram_user_id
        ).first()
        
        if existing_vote:
            # Check if user has exceeded change limit
            if existing_vote.vote_changes_count >= 2:
                return {
                    "success": False,
                    "message": "No vote changes left! (2/2 used)",
                    "changes_left": 0
                }
            
            # Check if trying to vote the same thing
            if existing_vote.current_vote == vote_type:
                return {
                    "success": False,
                    "message": "You already voted this way!",
                    "changes_left": 2 - existing_vote.vote_changes_count
                }
            
            # Update existing vote
            existing_vote.current_vote = vote_type
            existing_vote.vote_changes_count += 1
            existing_vote.last_vote_at = datetime.utcnow()
            
            changes_left = 2 - existing_vote.vote_changes_count
            vote_text = "👍 Liked!" if vote_type == 1 else "👎 Disliked!"
            
            if changes_left == 0:
                message = f"{vote_text} (Last change!)"
            else:
                message = f"{vote_text} ({changes_left} changes left)"
            
            db.commit()
            return {
                "success": True,
                "message": message,
                "changes_left": changes_left
            }
        else:
            # Create new vote
            new_vote = UserVotes(
                raw_data_id=raw_data_id,
                telegram_user_id=telegram_user_id,
                current_vote=vote_type,
                vote_changes_count=0
            )
            db.add(new_vote)
            db.commit()
            
            vote_text = "👍 Liked!" if vote_type == 1 else "👎 Disliked!"
            return {
                "success": True,
                "message": f"{vote_text} (2 changes available)",
                "changes_left": 2
            }
        
    except Exception as e:
        logger.error(f"Error handling user vote: {e}")
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
        # Get all votes for this question
        votes = db.query(UserVotes).filter(UserVotes.raw_data_id == raw_data_id).all()
        
        likes = sum(1 for vote in votes if vote.current_vote == 1)
        dislikes = sum(1 for vote in votes if vote.current_vote == -1)
        
        return {
            "likes": likes,
            "dislikes": dislikes,
            "total_votes": len(votes),
            "score": likes - dislikes
        }
    except Exception as e:
        logger.error(f"Error getting vote statistics: {e}")
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
        return {}

def init_db():
    """
    Initialize database and create all tables.
    """
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Create indexes for performance
        with engine.connect() as conn:
            # Raw data indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_raw_data_telegram_id ON raw_data(telegram_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_raw_data_language ON raw_data(language);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_raw_data_created_at ON raw_data(created_at);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_raw_data_is_duplicate ON raw_data(is_duplicate);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_raw_data_admin_approved ON raw_data(admin_approved);"))
            
            # Training data indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_training_data_source_id ON training_data(source_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_training_data_language ON training_data(language);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_training_data_is_active ON training_data(is_active);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_training_data_is_answer_duplicate ON training_data(is_answer_duplicate);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_training_data_review_status ON training_data(review_status);"))
            
            # User analytics indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_analytics_telegram_id ON user_analytics(telegram_id);"))
            
            # User votes indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_votes_raw_data_id ON user_votes(raw_data_id);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_user_votes_telegram_user_id ON user_votes(telegram_user_id);"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS idx_user_votes_unique ON user_votes(raw_data_id, telegram_user_id);"))
            
            # System metrics indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);"))
            
            conn.commit()
        
        logger.info("SQLite database initialized successfully with all tables and indexes!")
        print("✅ Database initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def get_db() -> Session:
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
    'get_database_statistics'
]
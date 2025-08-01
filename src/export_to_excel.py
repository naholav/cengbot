#!/usr/bin/env python3
"""
Raw Database to Excel Export Script for CengBot
===============================================

This script exports ONLY the raw_data table to Excel format with timestamp information.
Exports raw user interactions data for analysis and backup purposes.

Author: naholav
"""

import os
import sys
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

# Add parent directory to path to import database models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database_models import RawData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'university_bot.db')}"
EXCEL_DIR = os.path.join(BASE_DIR, 'excel')

def get_database_connection():
    """Create database connection and session."""
    try:
        engine = create_engine(DATABASE_URL, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return engine, SessionLocal()
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def export_raw_data(db_session):
    """Export raw_data table to DataFrame."""
    try:
        query = db_session.query(RawData)
        data = []
        
        for record in query.all():
            data.append({
                'id': record.id,
                'telegram_id': record.telegram_id,
                'telegram_message_id': record.telegram_message_id,
                'username': record.username,
                'question': record.question,
                'answer': record.answer,
                'language': record.language,
                'like': record.like,
                'admin_approved': record.admin_approved,
                'is_duplicate': record.is_duplicate,
                'duplicate_of_id': record.duplicate_of_id,
                'similarity_score': record.similarity_score,
                'created_at': record.created_at,
                'answered_at': record.answered_at,
                'message_thread_id': record.message_thread_id,
                'processing_time': record.processing_time,
                'model_version': record.model_version,
                'context_length': record.context_length,
                'response_length': record.response_length,
                'quality_score': record.quality_score,
                'sentiment_score': record.sentiment_score,
                'complexity_score': record.complexity_score,
                'topic_category': record.topic_category,
                'keywords': record.keywords,
                'admin_notes': record.admin_notes,
                'last_updated': record.last_updated
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error exporting raw_data: {e}")
        return pd.DataFrame()

def export_training_data(db_session):
    """Export training_data table to DataFrame."""
    try:
        query = db_session.query(TrainingData)
        data = []
        
        for record in query.all():
            data.append({
                'id': record.id,
                'source_id': record.source_id,
                'question': record.question,
                'answer': record.answer,
                'language': record.language,
                'is_answer_duplicate': record.is_answer_duplicate,
                'duplicate_answer_of_id': record.duplicate_answer_of_id,
                'answer_similarity_score': record.answer_similarity_score,
                'quality_score': record.quality_score,
                'usage_count': record.usage_count,
                'effectiveness_score': record.effectiveness_score,
                'category': record.category,
                'difficulty_level': record.difficulty_level,
                'topic_tags': record.topic_tags,
                'training_weight': record.training_weight,
                'augmentation_count': record.augmentation_count,
                'source_type': record.source_type,
                'review_status': record.review_status,
                'reviewer_id': record.reviewer_id,
                'review_date': record.review_date,
                'admin_notes': record.admin_notes,
                'version': record.version,
                'is_active': record.is_active,
                'created_at': record.created_at,
                'last_used_at': record.last_used_at,
                'last_updated': record.last_updated
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error exporting training_data: {e}")
        return pd.DataFrame()

def export_user_votes(db_session):
    """Export user_votes table to DataFrame."""
    try:
        query = db_session.query(UserVotes)
        data = []
        
        for record in query.all():
            data.append({
                'id': record.id,
                'raw_data_id': record.raw_data_id,
                'telegram_user_id': record.telegram_user_id,
                'current_vote': record.current_vote,
                'vote_changes_count': record.vote_changes_count,
                'first_vote_at': record.first_vote_at,
                'last_vote_at': record.last_vote_at,
                'created_at': record.created_at
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error exporting user_votes: {e}")
        return pd.DataFrame()

def export_user_analytics(db_session):
    """Export user_analytics table to DataFrame."""
    try:
        query = db_session.query(UserAnalytics)
        data = []
        
        for record in query.all():
            data.append({
                'id': record.id,
                'telegram_id': record.telegram_id,
                'session_count': record.session_count,
                'total_questions': record.total_questions,
                'total_likes': record.total_likes,
                'total_dislikes': record.total_dislikes,
                'preferred_language': record.preferred_language,
                'most_active_hour': record.most_active_hour,
                'avg_session_duration': record.avg_session_duration,
                'avg_satisfaction': record.avg_satisfaction,
                'engagement_score': record.engagement_score,
                'retention_score': record.retention_score,
                'first_interaction': record.first_interaction,
                'last_interaction': record.last_interaction,
                'created_at': record.created_at,
                'updated_at': record.updated_at
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error exporting user_analytics: {e}")
        return pd.DataFrame()

def export_system_metrics(db_session):
    """Export system_metrics table to DataFrame."""
    try:
        query = db_session.query(SystemMetrics)
        data = []
        
        for record in query.all():
            data.append({
                'id': record.id,
                'metric_name': record.metric_name,
                'metric_value': record.metric_value,
                'metric_type': record.metric_type,
                'service_name': record.service_name,
                'instance_id': record.instance_id,
                'tags': record.tags,
                'timestamp': record.timestamp,
                'details': record.details
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error exporting system_metrics: {e}")
        return pd.DataFrame()

def main():
    """Main export function."""
    try:
        # Ensure excel directory exists
        os.makedirs(EXCEL_DIR, exist_ok=True)
        
        # Get database connection
        engine, db_session = get_database_connection()
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cengbot_database_export_{timestamp}.xlsx"
        filepath = os.path.join(EXCEL_DIR, filename)
        
        logger.info(f"Starting database export to {filepath}")
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Export ONLY raw_data table
            logger.info("Exporting raw_data table...")
            raw_data_df = export_raw_data(db_session)
            if not raw_data_df.empty:
                raw_data_df.to_excel(writer, sheet_name='raw_data', index=False)
                logger.info(f"Exported {len(raw_data_df)} raw_data records")
            else:
                logger.warning("No raw_data records found to export")
            
            # Create summary sheet
            summary_data = {
                'Export Information': ['Export Date', 'Export Time', 'Database File', 'Tables Exported'],
                'Values': [
                    datetime.now().strftime("%Y-%m-%d"),
                    datetime.now().strftime("%H:%M:%S"),
                    'university_bot.db',
                    'raw_data only'
                ]
            }
            
            table_summary = {
                'Table Name': ['raw_data'],
                'Record Count': [len(raw_data_df)],
                'Description': ['User interactions and questions']
            }
            
            summary_df = pd.DataFrame(summary_data)
            table_summary_df = pd.DataFrame(table_summary)
            
            # Write summary to first sheet
            summary_df.to_excel(writer, sheet_name='export_summary', index=False, startrow=0)
            table_summary_df.to_excel(writer, sheet_name='export_summary', index=False, startrow=6)
            
            logger.info("Summary sheet created")
        
        # Close database session
        db_session.close()
        
        logger.info(f"✅ Raw database export completed successfully!")
        logger.info(f"📁 File saved: {filepath}")
        logger.info(f"📊 Total records exported: {len(raw_data_df)}")
        
        print(f"✅ Raw database export completed successfully!")
        print(f"📁 File: {filename}")
        print(f"📍 Location: {filepath}")
        print(f"📊 Records: {len(raw_data_df)} raw_data entries")
        
        return True
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        print(f"❌ Export failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
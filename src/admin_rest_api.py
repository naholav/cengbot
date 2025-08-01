from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Generic, TypeVar
from datetime import datetime
from sqlalchemy import func, text
from database_models import SessionLocal, RawData, TrainingData, mark_duplicate_questions, mark_duplicate_answers, get_vote_statistics
from error_handler import handle_database_error, handle_api_error, ErrorLevel
import uvicorn
import os
import psutil
import torch
import math
import secrets
import glob
import hashlib
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.env_loader import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

# Load environment variables
config = load_config()

# Security
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    if not ADMIN_PASSWORD_HASH:
        # For development/first run, use default password
        correct_password = secrets.compare_digest(credentials.password, "cucengedutr")
        correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    else:
        # Production mode with hashed password
        password_hash = hash_password(credentials.password)
        correct_password = secrets.compare_digest(password_hash, ADMIN_PASSWORD_HASH)
        correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

app = FastAPI(title="University Bot Admin API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class RawDataResponse(BaseModel):
    id: int
    telegram_id: int
    telegram_message_id: Optional[int]
    username: Optional[str]
    question: str
    answer: Optional[str]
    language: Optional[str]
    like: Optional[int]
    admin_approved: int
    is_duplicate: bool
    duplicate_of_id: Optional[int]
    similarity_score: Optional[float]
    created_at: datetime
    answered_at: Optional[datetime]
    message_thread_id: Optional[int]
    likes: Optional[int] = 0
    dislikes: Optional[int] = 0
    total_votes: Optional[int] = 0
    vote_score: Optional[int] = 0
    
    class Config:
        from_attributes = True

class TrainingDataResponse(BaseModel):
    id: int
    source_id: Optional[int]  # Can be NULL for standalone training data
    question: str
    answer: str
    language: Optional[str]
    created_at: datetime
    point: Optional[int]  # From source raw_data
    is_answer_duplicate: Optional[bool]
    duplicate_answer_of_id: Optional[int]
    answer_similarity_score: Optional[float]
    
    class Config:
        from_attributes = True

class UpdateAnswerRequest(BaseModel):
    answer: str

class ApproveResponse(BaseModel):
    success: bool
    message: str
    training_data_id: Optional[int]

class PaginatedResponse(BaseModel, Generic[T]):
    data: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

@app.get("/")
def root():
    return {"message": "University Bot Admin API"}

@app.get("/raw-data", response_model=PaginatedResponse[RawDataResponse])
def get_raw_data(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    only_unapproved: bool = False
):
    """Get all raw data with page-based pagination"""
    db = SessionLocal()
    try:
        query = db.query(RawData)
        
        if only_unapproved:
            query = query.filter(RawData.admin_approved == 0)
        
        total = query.count()
        total_pages = math.ceil(total / page_size)
        skip = (page - 1) * page_size
        
        raw_data = query.order_by(RawData.created_at.desc()).offset(skip).limit(page_size).all()
        
        # Add vote statistics to each item
        data = []
        for item in raw_data:
            vote_stats = get_vote_statistics(db, item.id)
            item_dict = {
                "id": item.id,
                "telegram_id": item.telegram_id,
                "telegram_message_id": item.telegram_message_id,
                "username": item.username,
                "question": item.question,
                "answer": item.answer,
                "language": item.language,
                "like": item.like,
                "admin_approved": item.admin_approved,
                "is_duplicate": item.is_duplicate,
                "duplicate_of_id": item.duplicate_of_id,
                "similarity_score": item.similarity_score,
                "created_at": item.created_at,
                "answered_at": item.answered_at,
                "message_thread_id": item.message_thread_id,
                "likes": vote_stats["likes"],
                "dislikes": vote_stats["dislikes"],
                "total_votes": vote_stats["total_votes"],
                "vote_score": vote_stats["score"]
            }
            data.append(RawDataResponse(**item_dict))
        
        return PaginatedResponse(
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
    except Exception as e:
        error_response = handle_database_error(e, "select", "raw_data")
        raise HTTPException(status_code=500, detail=error_response["error"]["message"])
    finally:
        db.close()

@app.put("/raw-data/{item_id}")
def update_answer(item_id: int, request: UpdateAnswerRequest):
    """Update answer for a raw data entry"""
    db = SessionLocal()
    try:
        raw_data = db.query(RawData).filter(RawData.id == item_id).first()
        if not raw_data:
            raise HTTPException(status_code=404, detail="Item not found")
        
        raw_data.answer = request.answer
        raw_data.answered_at = datetime.utcnow()
        db.commit()
        
        return {"success": True, "message": "Answer updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        error_response = handle_database_error(e, "update", "raw_data")
        raise HTTPException(status_code=500, detail=error_response["error"]["message"])
    finally:
        db.close()

@app.post("/approve/{item_id}", response_model=ApproveResponse)
def approve_to_training(item_id: int):
    """Approve raw data and copy to training data with duplicate detection"""
    db = SessionLocal()
    try:
        # Get raw data
        raw_data = db.query(RawData).filter(RawData.id == item_id).first()
        if not raw_data:
            raise HTTPException(status_code=404, detail="Item not found")
        
        if not raw_data.answer:
            raise HTTPException(status_code=400, detail="Cannot approve without answer")
        
        # Check if already in training data
        existing = db.query(TrainingData).filter(TrainingData.source_id == item_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Already in training data")
        
        # Create training data entry
        training_data = TrainingData(
            source_id=raw_data.id,
            question=raw_data.question,
            answer=raw_data.answer,
            language=raw_data.language
        )
        db.add(training_data)
        
        # Mark as approved
        raw_data.admin_approved = 1
        
        db.commit()
        db.refresh(training_data)
        
        # Check for duplicate questions in raw data with optimized threshold
        mark_duplicate_questions(db, raw_data.id, raw_data.question)
        
        # Check for duplicate answers in training data with optimized threshold
        mark_duplicate_answers(db, training_data.id, training_data.answer)
        
        return ApproveResponse(
            success=True,
            message="Successfully approved to training data with duplicate detection",
            training_data_id=training_data.id
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/stats")
def get_statistics():
    """Get basic statistics"""
    db = SessionLocal()
    try:
        total_questions = db.query(RawData).count()
        approved_questions = db.query(RawData).filter(RawData.admin_approved == 1).count()
        liked_questions = db.query(RawData).filter(RawData.like == 1).count()
        disliked_questions = db.query(RawData).filter(RawData.like == -1).count()
        training_data_count = db.query(TrainingData).count()
        duplicate_count = db.query(RawData).filter(RawData.is_duplicate == True).count()
        
        return {
            "total_questions": total_questions,
            "approved_questions": approved_questions,
            "liked_questions": liked_questions,
            "disliked_questions": disliked_questions,
            "training_data_count": training_data_count,
            "duplicate_count": duplicate_count,
            "approval_rate": f"{(approved_questions / total_questions * 100):.1f}%" if total_questions > 0 else "0%"
        }
    finally:
        db.close()

@app.get("/duplicates")
def get_duplicate_groups():
    """Get duplicate question groups"""
    db = SessionLocal()
    try:
        # Get all questions that are originals of duplicates
        originals = db.query(RawData).filter(
            RawData.id.in_(
                db.query(RawData.duplicate_of_id).filter(RawData.duplicate_of_id.isnot(None))
            )
        ).all()
        
        groups = []
        for original in originals:
            duplicates = db.query(RawData).filter(RawData.duplicate_of_id == original.id).all()
            groups.append({
                "original": RawDataResponse.from_orm(original),
                "duplicates": [RawDataResponse.from_orm(d) for d in duplicates]
            })
        
        return groups
    finally:
        db.close()

@app.get("/training-data", response_model=PaginatedResponse[TrainingDataResponse])
def get_training_data(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Get all training data with page-based pagination"""
    db = SessionLocal()
    try:
        # Use LEFT JOIN to handle NULL source_id values
        query = db.query(
            TrainingData.id,
            TrainingData.source_id,
            TrainingData.question,
            TrainingData.answer,
            TrainingData.language,
            TrainingData.created_at,
            TrainingData.is_answer_duplicate,
            TrainingData.duplicate_answer_of_id,
            TrainingData.answer_similarity_score,
            RawData.like.label('point')
        ).outerjoin(RawData, TrainingData.source_id == RawData.id)
        
        total = query.count()
        total_pages = math.ceil(total / page_size)
        skip = (page - 1) * page_size
        
        data = query.order_by(TrainingData.created_at.desc()).offset(skip).limit(page_size).all()
        
        # Convert to response model
        training_data = [
            TrainingDataResponse(
                id=item.id,
                source_id=item.source_id,  # Can be None for standalone training data
                question=item.question,
                answer=item.answer,
                language=item.language,
                created_at=item.created_at,
                point=item.point,  # Will be None if no raw_data match
                is_answer_duplicate=item.is_answer_duplicate,
                duplicate_answer_of_id=item.duplicate_answer_of_id,
                answer_similarity_score=item.answer_similarity_score
            )
            for item in data
        ]
        
        return PaginatedResponse(
            data=training_data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
    finally:
        db.close()

@app.delete("/raw-data/{item_id}")
def delete_raw_data(item_id: int):
    """Delete raw data entry"""
    db = SessionLocal()
    try:
        # Get raw data
        raw_data = db.query(RawData).filter(RawData.id == item_id).first()
        if not raw_data:
            raise HTTPException(status_code=404, detail="Raw data not found")
        
        # Check if it has associated training data
        training_data = db.query(TrainingData).filter(TrainingData.source_id == item_id).first()
        if training_data:
            # Delete training data first
            db.delete(training_data)
        
        # Delete raw data
        db.delete(raw_data)
        db.commit()
        
        return {"success": True, "message": "Raw data deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.delete("/training-data/{item_id}")
def delete_training_data(item_id: int):
    """Remove from training data"""
    db = SessionLocal()
    try:
        # Get training data
        training_data = db.query(TrainingData).filter(TrainingData.id == item_id).first()
        if not training_data:
            raise HTTPException(status_code=404, detail="Training data not found")
        
        # Update raw_data admin_approved status
        raw_data = db.query(RawData).filter(RawData.id == training_data.source_id).first()
        if raw_data:
            raw_data.admin_approved = 0
        
        # Delete training data
        db.delete(training_data)
        db.commit()
        
        return {"success": True, "message": "Training data removed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.post("/detect-duplicates")
def detect_duplicates():
    """Detect duplicates in existing raw data and training data"""
    db = SessionLocal()
    try:
        # Clear existing duplicate markings first
        db.query(RawData).update({
            RawData.is_duplicate: False,
            RawData.duplicate_of_id: None,
            RawData.similarity_score: None
        })
        
        db.query(TrainingData).update({
            TrainingData.is_answer_duplicate: False,
            TrainingData.duplicate_answer_of_id: None,
            TrainingData.answer_similarity_score: None
        })
        
        db.commit()
        
        # Process raw data for duplicate questions
        raw_data_entries = db.query(RawData).all()
        question_duplicates_found = 0
        
        logger.info(f"Starting duplicate detection for {len(raw_data_entries)} raw data entries")
        
        for entry in raw_data_entries:
            if entry.question and len(entry.question.strip()) > 10:  # Only process meaningful questions
                was_duplicate = mark_duplicate_questions(db, entry.id, entry.question)
                if was_duplicate:
                    question_duplicates_found += 1
        
        # Process training data for duplicate answers
        training_data_entries = db.query(TrainingData).all()
        answer_duplicates_found = 0
        
        logger.info(f"Starting duplicate detection for {len(training_data_entries)} training data entries")
        
        for entry in training_data_entries:
            if entry.answer and len(entry.answer.strip()) > 10:  # Only process meaningful answers
                was_duplicate = mark_duplicate_answers(db, entry.id, entry.answer)
                if was_duplicate:
                    answer_duplicates_found += 1
        
        return {
            "success": True,
            "message": f"Duplicate detection completed. Found {question_duplicates_found} question duplicates and {answer_duplicates_found} answer duplicates",
            "question_duplicates": question_duplicates_found,
            "answer_duplicates": answer_duplicates_found
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error in duplicate detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/health")
def health_check():
    """System health check endpoint"""
    try:
        # Basic system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_info = {}
        if gpu_available:
            gpu_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            }
        
        # Database connectivity
        db_healthy = True
        try:
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
        except Exception:
            db_healthy = False
        
        # Check if model files exist
        model_available = os.path.exists("/home/ceng/cu_ceng_bot/models/active-model")
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "system": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "gpu": {
                "available": gpu_available,
                **gpu_info
            },
            "database": {
                "healthy": db_healthy
            },
            "model": {
                "available": model_available
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        }

@app.post("/auth/login")
def login(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify login credentials"""
    username = verify_credentials(credentials)
    return {"success": True, "username": username}

@app.get("/docs/list")
def list_documents(_: str = Depends(verify_credentials)):
    """List available documentation files"""
    docs = []
    
    # Get README files, excluding node_modules
    readme_files = glob.glob("/home/ceng/cu_ceng_bot/**/*.md", recursive=True)
    readme_files.extend(glob.glob("/home/ceng/cu_ceng_bot/**/README", recursive=True))
    
    # Filter out node_modules and other unwanted directories
    readme_files = [f for f in readme_files if 'node_modules' not in f and '.git' not in f]
    
    # Get log files
    log_files = glob.glob("/home/ceng/cu_ceng_bot/logs/*.log")
    
    # Get script documentation
    script_files = glob.glob("/home/ceng/cu_ceng_bot/scripts/*.sh")
    
    # Format the file list
    for file in readme_files:
        rel_path = os.path.relpath(file, "/home/ceng/cu_ceng_bot")
        docs.append({
            "path": rel_path,
            "name": os.path.basename(file),
            "type": "documentation",
            "size": os.path.getsize(file)
        })
    
    for file in log_files:
        rel_path = os.path.relpath(file, "/home/ceng/cu_ceng_bot")
        docs.append({
            "path": rel_path,
            "name": os.path.basename(file),
            "type": "log",
            "size": os.path.getsize(file)
        })
    
    for file in script_files:
        rel_path = os.path.relpath(file, "/home/ceng/cu_ceng_bot")
        docs.append({
            "path": rel_path,
            "name": os.path.basename(file),
            "type": "script",
            "size": os.path.getsize(file)
        })
    
    return sorted(docs, key=lambda x: x["type"])

@app.get("/docs/read", response_class=PlainTextResponse)
def read_document(path: str, _: str = Depends(verify_credentials)):
    """Read a specific document file"""
    # Security: prevent path traversal
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")
    
    full_path = os.path.join("/home/ceng/cu_ceng_bot", path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=400, detail="Not a file")
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
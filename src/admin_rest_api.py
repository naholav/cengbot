from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
from datetime import datetime
from sqlalchemy import func, text
from database_models import SessionLocal, RawData, TrainingData
import uvicorn

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
    created_at: datetime
    answered_at: Optional[datetime]
    message_thread_id: Optional[int]
    
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
    
    class Config:
        from_attributes = True

class UpdateAnswerRequest(BaseModel):
    answer: str

class ApproveResponse(BaseModel):
    success: bool
    message: str
    training_data_id: Optional[int]

@app.get("/")
def root():
    return {"message": "University Bot Admin API"}

@app.get("/raw-data", response_model=List[RawDataResponse])
def get_raw_data(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    only_unapproved: bool = False
):
    """Get all raw data with pagination"""
    db = SessionLocal()
    try:
        query = db.query(RawData)
        
        if only_unapproved:
            query = query.filter(RawData.admin_approved == 0)
        
        total = query.count()
        data = query.order_by(RawData.created_at.desc()).offset(skip).limit(limit).all()
        
        return data
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
    finally:
        db.close()

@app.post("/approve/{item_id}", response_model=ApproveResponse)
def approve_to_training(item_id: int):
    """Approve raw data and copy to training data"""
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
        
        return ApproveResponse(
            success=True,
            message="Successfully approved to training data",
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

@app.get("/training-data", response_model=List[TrainingDataResponse])
def get_training_data(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get all training data with pagination"""
    db = SessionLocal()
    try:
        # Use LEFT JOIN to handle NULL source_id values
        query = db.query(
            TrainingData.id,
            TrainingData.source_id,
            TrainingData.question,
            TrainingData.answer,
            TrainingData.created_at,
            RawData.like.label('point')
        ).outerjoin(RawData, TrainingData.source_id == RawData.id)
        
        total = query.count()
        data = query.order_by(TrainingData.created_at.desc()).offset(skip).limit(limit).all()
        
        # Convert to response model
        return [
            TrainingDataResponse(
                id=item.id,
                source_id=item.source_id,  # Can be None for standalone training data
                question=item.question,
                answer=item.answer,
                language=None,  # Add language field (currently not stored in these records)
                created_at=item.created_at,
                point=item.point  # Will be None if no raw_data match
            )
            for item in data
        ]
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
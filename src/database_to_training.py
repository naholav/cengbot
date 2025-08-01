#!/usr/bin/env python3
"""
Database to Training Data Converter
==================================

This module converts database training data to JSONL format for model training.
It automatically exports approved training data from the database to the format
required by the training script.

Features:
- Automatic database export to JSONL format
- Language mapping (TR -> turkish, EN -> english)
- Data validation and filtering
- Backup creation
- Progress tracking

Author: naholav
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Import database models
from database_models import SessionLocal, TrainingData, RawData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseToTrainingConverter:
    """Convert database training data to JSONL format for training"""
    
    def __init__(self, base_path: str = "/home/ceng/cu_ceng_bot"):
        self.base_path = Path(base_path)
        self.data_dir = self.base_path / "data"
        self.backup_dir = self.base_path / "data" / "backups"
        self.training_file = self.data_dir / "cengbot_qa_augmented.jsonl"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
    
    def map_language(self, db_language: str) -> str:
        """Map database language codes to training format"""
        language_map = {
            'TR': 'turkish',
            'EN': 'english',
            'turkish': 'turkish',
            'english': 'english'
        }
        return language_map.get(db_language, 'english')
    
    def validate_training_data(self, data: Dict) -> bool:
        """Validate training data entry"""
        required_fields = ['id', 'question', 'answer', 'language']
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
            
            if not data[field]:
                logger.warning(f"Empty value for field: {field}")
                return False
        
        # Check minimum content length
        if len(data['question'].strip()) < 3:
            logger.warning(f"Question too short: {data['question']}")
            return False
            
        if len(data['answer'].strip()) < 5:
            logger.warning(f"Answer too short: {data['answer']}")
            return False
        
        return True
    
    def backup_existing_file(self) -> Optional[Path]:
        """Create backup of existing training file"""
        if not self.training_file.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"cengbot_qa_augmented_backup_{timestamp}.jsonl"
        
        try:
            import shutil
            shutil.copy2(self.training_file, backup_file)
            logger.info(f"Created backup: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def export_from_database(self, include_inactive: bool = False) -> List[Dict]:
        """Export training data from database"""
        db = SessionLocal()
        exported_data = []
        
        try:
            # Query training data
            query = db.query(TrainingData)
            
            # Filter active records only (unless specified)
            if not include_inactive:
                query = query.filter(TrainingData.is_active == True)
            
            # Order by creation date
            query = query.order_by(TrainingData.created_at)
            
            training_records = query.all()
            
            logger.info(f"Found {len(training_records)} training records in database")
            
            for record in training_records:
                # Convert database record to training format
                training_entry = {
                    "id": record.id,
                    "question": record.question.strip(),
                    "answer": record.answer.strip(),
                    "language": self.map_language(record.language)
                }
                
                # Validate data
                if self.validate_training_data(training_entry):
                    exported_data.append(training_entry)
                else:
                    logger.warning(f"Skipping invalid training record ID: {record.id}")
            
            logger.info(f"Successfully exported {len(exported_data)} valid training entries")
            
        except Exception as e:
            logger.error(f"Error exporting from database: {e}")
            raise
        finally:
            db.close()
        
        return exported_data
    
    def load_existing_data(self) -> List[Dict]:
        """Load existing training data from JSONL file"""
        if not self.training_file.exists():
            return []
        
        existing_data = []
        try:
            with open(self.training_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_data.append(json.loads(line))
            
            logger.info(f"Loaded {len(existing_data)} existing training entries")
            return existing_data
            
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            return []

    def save_to_jsonl(self, data: List[Dict], file_path: Optional[Path] = None, append_mode: bool = True) -> Path:
        """Save training data to JSONL file"""
        if file_path is None:
            file_path = self.training_file
        
        try:
            if append_mode:
                # Load existing data first
                existing_data = self.load_existing_data()
                
                # Get existing IDs to avoid duplicates
                existing_ids = set()
                for entry in existing_data:
                    existing_ids.add(entry.get('id'))
                
                # Filter out duplicates from new data
                new_data = []
                for entry in data:
                    if entry.get('id') not in existing_ids:
                        new_data.append(entry)
                
                # Combine existing and new data
                all_data = existing_data + new_data
                logger.info(f"Appending {len(new_data)} new entries to {len(existing_data)} existing entries")
            else:
                all_data = data
            
            # Save all data
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in all_data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"Saved {len(all_data)} total training entries to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving to JSONL: {e}")
            raise
    
    def get_export_statistics(self, data: List[Dict]) -> Dict:
        """Get statistics about exported data"""
        if not data:
            return {
                "total_entries": 0,
                "by_language": {},
                "avg_question_length": 0,
                "avg_answer_length": 0
            }
        
        # Count by language
        language_counts = {}
        total_question_length = 0
        total_answer_length = 0
        
        for entry in data:
            lang = entry['language']
            language_counts[lang] = language_counts.get(lang, 0) + 1
            total_question_length += len(entry['question'])
            total_answer_length += len(entry['answer'])
        
        return {
            "total_entries": len(data),
            "by_language": language_counts,
            "avg_question_length": total_question_length / len(data),
            "avg_answer_length": total_answer_length / len(data)
        }
    
    def full_export(self, include_inactive: bool = False, create_backup: bool = True) -> Dict:
        """Complete export process from database to training file"""
        logger.info("Starting full database to training data export...")
        
        # Create backup if requested
        backup_file = None
        if create_backup:
            backup_file = self.backup_existing_file()
        
        # Export from database
        training_data = self.export_from_database(include_inactive=include_inactive)
        
        if not training_data:
            logger.warning("No training data found in database")
            return {
                "success": False,
                "message": "No training data found in database",
                "statistics": {}
            }
        
        # Save to JSONL file (append mode by default)
        output_file = self.save_to_jsonl(training_data, append_mode=True)
        
        # Get statistics
        statistics = self.get_export_statistics(training_data)
        
        # Return results
        result = {
            "success": True,
            "message": f"Successfully exported {len(training_data)} training entries",
            "output_file": str(output_file),
            "backup_file": str(backup_file) if backup_file else None,
            "statistics": statistics
        }
        
        logger.info("Export completed successfully!")
        return result

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export database training data to JSONL format")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive training records")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup file")
    parser.add_argument("--output", type=str, help="Output file path (default: data/cengbot_qa_augmented.jsonl)")
    
    args = parser.parse_args()
    
    # Create converter
    converter = DatabaseToTrainingConverter()
    
    # Export data
    result = converter.full_export(
        include_inactive=args.include_inactive,
        create_backup=not args.no_backup
    )
    
    # Print results
    print(f"\n{'='*50}")
    print("DATABASE TO TRAINING DATA EXPORT RESULTS")
    print(f"{'='*50}")
    
    if result["success"]:
        print(f"✅ {result['message']}")
        print(f"📁 Output file: {result['output_file']}")
        if result['backup_file']:
            print(f"💾 Backup file: {result['backup_file']}")
        
        stats = result['statistics']
        print(f"\n📊 Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  By language: {stats['by_language']}")
        print(f"  Avg question length: {stats['avg_question_length']:.1f} chars")
        print(f"  Avg answer length: {stats['avg_answer_length']:.1f} chars")
    else:
        print(f"❌ {result['message']}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
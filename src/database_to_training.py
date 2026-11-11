#!/usr/bin/env python3
"""
Database to Training Data Converter
==================================

Exports the question/answer pairs that an admin approved into the
``training_data`` table to the JSONL format read by ``train_model.py``.

Which rows are exported:
- ``is_active`` rows only (``--include-inactive`` to include deactivated ones)
- never ``review_status == 'rejected'`` (NULL / 'pending' / 'approved' are fine:
  the admin panel's "approve to training" action is the approval itself; it sets
  ``review_status='approved'``, but rows approved by older versions may still be
  'pending')
- ``--approved-only``: only ``review_status == 'approved'``
- ``--exclude-disliked``: skip rows whose source chat answer was net-disliked by
  users (``raw_data.like == -1``) unless the admin edited the answer

JSONL record format (one object per line, UTF-8, ``ensure_ascii=False``)::

    {"id": 12, "question": "...", "answer": "...", "language": "turkish",
     "group_id": "db-12", "source": "database"}

``group_id`` ties an original to its augmented variants (written by
``data_augmentation.py``) so the training split never leaks variants of the
same question into both train and validation.

Default mode merges into the existing file: entries that came from the
database (``source`` database/augmented) are refreshed - rows that were
deleted, deactivated or rejected in the meantime disappear together with their
augmented variants - and legacy entries without provenance are kept. Content
(language + question + answer) is the dedup key, not ``id``. Writes are atomic.

Author: naholav
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sqlalchemy import or_  # noqa: E402

from database_models import RawData, SessionLocal, TrainingData  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_NAME = "cengbot_qa_augmented.jsonl"
MAX_BACKUPS = 10

_LANGUAGE_MAP = {
    "tr": "turkish",
    "turkish": "turkish",
    "türkçe": "turkish",
    "turkce": "turkish",
    "en": "english",
    "english": "english",
    "ingilizce": "english",
}
_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")


_DOTLESS_FOLD = str.maketrans({"İ": "i", "I": "i", "ı": "i"})


def _norm(text: str) -> str:
    """Case/whitespace-insensitive key; Turkish dotted/dotless i folded together."""
    return " ".join(str(text).translate(_DOTLESS_FOLD).casefold().split())


def content_key(entry: Dict) -> tuple:
    """Dedup key: same language + same question + same answer (normalized)."""
    return (entry.get("language"), _norm(entry.get("question", "")), _norm(entry.get("answer", "")))


def db_group_id(training_id: int) -> str:
    return f"db-{training_id}"


class DatabaseToTrainingConverter:
    """Convert database training data to JSONL format for training"""

    def __init__(self, base_path: Optional[str] = None, output_file: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else PROJECT_ROOT
        self.data_dir = self.base_path / "data"
        self.backup_dir = self.data_dir / "backups"
        self.training_file = Path(output_file) if output_file else self.data_dir / DEFAULT_OUTPUT_NAME

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.training_file.parent.mkdir(parents=True, exist_ok=True)

    def map_language(self, db_language: Optional[str], text: Optional[str] = None) -> str:
        """Map database language codes ('TR'/'EN'/...) to 'turkish'/'english'.

        Unknown/missing codes are guessed from Turkish characters in ``text``.
        """
        if isinstance(db_language, str):
            mapped = _LANGUAGE_MAP.get(db_language.strip().lower())
            if mapped:
                return mapped
        if text and any(ch in _TURKISH_CHARS for ch in text):
            return "turkish"
        return "english"

    def validate_training_data(self, data: Dict) -> bool:
        """Validate training data entry"""
        for field in ("id", "question", "answer", "language"):
            if field not in data or data[field] in (None, ""):
                logger.warning("Missing/empty required field: %s", field)
                return False
        if not isinstance(data["question"], str) or not isinstance(data["answer"], str):
            return False
        if len(data["question"].strip()) < 3:
            logger.warning("Question too short (id=%s)", data.get("id"))
            return False
        if len(data["answer"].strip()) < 5:
            logger.warning("Answer too short (id=%s)", data.get("id"))
            return False
        return True

    def backup_existing_file(self) -> Optional[Path]:
        """Create backup of existing training file"""
        if not self.training_file.exists():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_file = self.backup_dir / f"{self.training_file.stem}_backup_{timestamp}{self.training_file.suffix}"
        try:
            shutil.copy2(self.training_file, backup_file)
            logger.info("Created backup: %s", backup_file)
        except Exception as e:
            logger.error("Failed to create backup: %s", e)
            return None
        # Keep the newest MAX_BACKUPS backups (training runs export every time).
        backups = sorted(self.backup_dir.glob(f"{self.training_file.stem}_backup_*{self.training_file.suffix}"))
        for old in backups[:-MAX_BACKUPS]:
            try:
                old.unlink()
            except OSError:
                pass
        return backup_file

    def export_from_database(
        self,
        include_inactive: bool = False,
        approved_only: bool = False,
        exclude_disliked: bool = False,
    ) -> List[Dict]:
        """Export trainable rows from ``training_data`` (see module docstring)."""
        db = SessionLocal()
        exported: List[Dict] = []
        try:
            query = db.query(TrainingData, RawData).outerjoin(RawData, TrainingData.source_id == RawData.id)
            if not include_inactive:
                query = query.filter(TrainingData.is_active.is_(True))
            if approved_only:
                query = query.filter(TrainingData.review_status == "approved")
            else:
                # NULL != 'rejected' is NULL in SQL, so NULL must be allowed explicitly.
                query = query.filter(
                    or_(TrainingData.review_status.is_(None), TrainingData.review_status != "rejected")
                )
            rows = query.order_by(TrainingData.created_at, TrainingData.id).all()
            logger.info("Found %d candidate training records in database", len(rows))

            seen = set()
            skipped = {"invalid": 0, "duplicate": 0, "disliked": 0}
            for record, raw in rows:
                question = (record.question or "").strip()
                answer = (record.answer or "").strip()
                entry = {
                    "id": record.id,
                    "question": question,
                    "answer": answer,
                    "language": self.map_language(record.language, question + " " + answer),
                    "group_id": db_group_id(record.id),
                    "source": "database",
                }
                if not self.validate_training_data(entry):
                    skipped["invalid"] += 1
                    continue
                if exclude_disliked and raw is not None and raw.like == -1 and _norm(raw.answer or "") == _norm(answer):
                    skipped["disliked"] += 1
                    continue
                key = content_key(entry)
                if key in seen:
                    skipped["duplicate"] += 1
                    continue
                seen.add(key)
                exported.append(entry)

            logger.info("Exported %d valid training entries (skipped: %s)", len(exported), skipped)
        except Exception as e:
            logger.error("Error exporting from database: %s", e)
            raise
        finally:
            db.close()
        return exported

    def load_existing_data(self, file_path: Optional[Path] = None) -> List[Dict]:
        """Load existing training data from a JSONL file (invalid lines are skipped)."""
        file_path = Path(file_path) if file_path else self.training_file
        if not file_path.exists():
            return []
        existing: List[Dict] = []
        invalid = 0
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if isinstance(obj, dict):
                    existing.append(obj)
        if invalid:
            logger.warning("Skipped %d invalid line(s) in %s", invalid, file_path)
        logger.info("Loaded %d existing training entries from %s", len(existing), file_path)
        return existing

    @staticmethod
    def write_jsonl_atomic(data: List[Dict], file_path: Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = file_path.with_name(f".{file_path.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, file_path)
        except BaseException:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            raise

    def merge_entries(self, existing: List[Dict], new_data: List[Dict]) -> List[Dict]:
        """Refresh DB-derived entries, keep legacy ones, dedup by content."""
        active_groups = {e.get("group_id") for e in new_data if e.get("group_id")}
        new_by_key = {content_key(e): e for e in new_data}
        merged: List[Dict] = []
        seen = set()
        dropped_stale = 0
        for entry in existing:
            source = entry.get("source")
            group = entry.get("group_id")
            if source == "database":
                continue  # re-exported fresh below
            if source == "augmented" and isinstance(group, str) and group.startswith("db-") and group not in active_groups:
                dropped_stale += 1  # its original row is gone/deactivated/rejected
                continue
            key = content_key(entry)
            if key in seen or key in new_by_key:
                continue
            seen.add(key)
            merged.append(entry)
        appended = 0
        for entry in new_data:
            key = content_key(entry)
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
            appended += 1
        logger.info(
            "Merged: %d kept from existing file, %d from database, %d stale augmented entries dropped",
            len(merged) - appended,
            appended,
            dropped_stale,
        )
        return merged

    def save_to_jsonl(self, data: List[Dict], file_path: Optional[Path] = None, append_mode: bool = True) -> Path:
        """Save training data to JSONL (merge with the existing file when ``append_mode``)."""
        file_path = Path(file_path) if file_path else self.training_file
        all_data = self.merge_entries(self.load_existing_data(file_path), data) if append_mode else list(data)
        self.write_jsonl_atomic(all_data, file_path)
        logger.info("Saved %d total training entries to %s", len(all_data), file_path)
        return file_path

    def get_export_statistics(self, data: List[Dict]) -> Dict:
        """Get statistics about exported data"""
        if not data:
            return {"total_entries": 0, "by_language": {}, "avg_question_length": 0, "avg_answer_length": 0}
        language_counts: Dict[str, int] = {}
        total_q = total_a = 0
        for entry in data:
            lang = entry["language"]
            language_counts[lang] = language_counts.get(lang, 0) + 1
            total_q += len(entry["question"])
            total_a += len(entry["answer"])
        return {
            "total_entries": len(data),
            "by_language": language_counts,
            "avg_question_length": total_q / len(data),
            "avg_answer_length": total_a / len(data),
        }

    def full_export(
        self,
        include_inactive: bool = False,
        create_backup: bool = True,
        approved_only: bool = False,
        exclude_disliked: bool = False,
        append_mode: bool = True,
    ) -> Dict:
        """Complete export process from database to training file"""
        logger.info("Starting database to training data export -> %s", self.training_file)
        training_data = self.export_from_database(
            include_inactive=include_inactive, approved_only=approved_only, exclude_disliked=exclude_disliked
        )
        if not training_data:
            logger.warning("No training data found in database - %s left unchanged", self.training_file)
            return {
                "success": False,
                "message": "No training data found in database",
                "output_file": str(self.training_file),
                "backup_file": None,
                "statistics": self.get_export_statistics([]),
            }

        backup_file = self.backup_existing_file() if create_backup else None
        output_file = self.save_to_jsonl(training_data, append_mode=append_mode)
        statistics = self.get_export_statistics(training_data)
        logger.info("Export completed successfully!")
        return {
            "success": True,
            "message": f"Successfully exported {len(training_data)} training entries",
            "output_file": str(output_file),
            "backup_file": str(backup_file) if backup_file else None,
            "statistics": statistics,
        }


def main(argv: Optional[List[str]] = None) -> int:
    """Main function for command-line usage"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Export database training data to JSONL format")
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive training records")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backup file")
    parser.add_argument("--output", type=str, help=f"Output file path (default: data/{DEFAULT_OUTPUT_NAME})")
    parser.add_argument("--approved-only", action="store_true", help="Only rows with review_status='approved'")
    parser.add_argument("--exclude-disliked", action="store_true",
                        help="Skip rows whose (unedited) answer was net-disliked by users")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite the output file instead of merging into it")
    parser.add_argument("--base-path", type=str, default=None, help="Project root (default: auto-detected)")
    args = parser.parse_args(argv)

    converter = DatabaseToTrainingConverter(base_path=args.base_path, output_file=args.output)
    result = converter.full_export(
        include_inactive=args.include_inactive,
        create_backup=not args.no_backup,
        approved_only=args.approved_only,
        exclude_disliked=args.exclude_disliked,
        append_mode=not args.replace,
    )

    print(f"\n{'=' * 50}")
    print("DATABASE TO TRAINING DATA EXPORT RESULTS")
    print(f"{'=' * 50}")
    if not result["success"]:
        print(f"❌ {result['message']}")
        return 1
    print(f"✅ {result['message']}")
    print(f"📁 Output file: {result['output_file']}")
    if result["backup_file"]:
        print(f"💾 Backup file: {result['backup_file']}")
    stats = result["statistics"]
    print("\n📊 Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  By language: {stats['by_language']}")
    print(f"  Avg question length: {stats['avg_question_length']:.1f} chars")
    print(f"  Avg answer length: {stats['avg_answer_length']:.1f} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# =============================================================================
# CengBot Database -> Training Data Export
# Exports training data from the database (DATABASE_URL, default
# university_bot.db) to data/cengbot_qa_augmented.jsonl via
# src/database_to_training.py, using the project venv. All options are passed
# through, e.g. --approved-only, --exclude-disliked, --replace, --output FILE,
# --include-inactive, --no-backup.
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXPORT_SCRIPT="$PROJECT_ROOT/src/database_to_training.py"
DATA_DIR="$PROJECT_ROOT/data"

show_help() {
    echo "CengBot Database to Training Data Export"
    echo
    echo "Usage: $(basename "$0") [database_to_training.py options]"
    echo
    if [[ -x "$PYTHON" && -f "$EXPORT_SCRIPT" ]]; then
        "$PYTHON" "$EXPORT_SCRIPT" --help 2>/dev/null | sed -n '/^options:/,$p' || true
    fi
}

read -r -d '' SAMPLE_PY <<'PY' || true
import json, sys
for i, line in enumerate(sys.stdin, 1):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    print(f"  {i}. ID: {d.get('id')}, Language: {d.get('language')}")
    print(f"     Q: {str(d.get('question', ''))[:50]}...")
    print(f"     A: {str(d.get('answer', ''))[:50]}...")
PY

# --output is relative to the caller's directory (abspath from _common.sh; the export runs in the project root)
ARGS=()
OUTPUT_FILE="$DATA_DIR/cengbot_qa_augmented.jsonl"
while (( $# > 0 )); do
    case "$1" in
        -h|--help) show_help; exit 0 ;;
        --output)
            [[ $# -ge 2 && -n "$2" ]] || die "--output needs a file"
            OUTPUT_FILE="$(abspath "$2")"; ARGS+=(--output "$OUTPUT_FILE"); shift ;;
        --output=*)
            [[ -n "${1#--output=}" ]] || die "--output needs a file"
            OUTPUT_FILE="$(abspath "${1#--output=}")"; ARGS+=(--output "$OUTPUT_FILE") ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

print_header "CengBot Database to Training Data Export"
require_python
[[ -f "$EXPORT_SCRIPT" ]] || die "Export script not found: $EXPORT_SCRIPT"

# Count active training rows through the ORM (honours DATABASE_URL)
print_status "Checking database..."
count=$(cd "$PROJECT_ROOT" && "$PYTHON" -c '
import sys
sys.path.insert(0, "src")
from sqlalchemy import or_
from database_models import SessionLocal, TrainingData
db = SessionLocal()
try:
    # same filter as database_to_training.py: active and not rejected (NULL = not rejected)
    print(db.query(TrainingData).filter(
        TrainingData.is_active == True,  # noqa: E712
        or_(TrainingData.review_status.is_(None), TrainingData.review_status != "rejected"),
    ).count())
finally:
    db.close()
' 2>/dev/null | tail -n 1) || count=""
if [[ ! "$count" =~ ^[0-9]+$ ]]; then
    die "Could not read the training_data table (is the database initialised? see logs)"
fi
if (( count == 0 )); then
    print_warning "No exportable training data (active, not rejected) in the database - nothing to export"
    exit 1
fi
print_success "Database contains $count exportable training records (active, not rejected)"

print_status "Running database export..."
( cd "$PROJECT_ROOT" && "$PYTHON" "$EXPORT_SCRIPT" "${ARGS[@]+"${ARGS[@]}"}" ) || die "Database export failed!"
print_success "Database export completed successfully!"

if [[ -f "$OUTPUT_FILE" ]]; then
    print_status "Training file: $OUTPUT_FILE"
    print_status "  Lines: $(wc -l < "$OUTPUT_FILE")    Size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    print_status "Sample training data:"
    head -n 3 -- "$OUTPUT_FILE" | "$PYTHON" -I -c "$SAMPLE_PY"
fi
print_success "Export process completed successfully!"

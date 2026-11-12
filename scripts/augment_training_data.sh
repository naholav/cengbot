#!/usr/bin/env bash

# CengBot Training Data Augmentation Script
# Augments training data using Anthropic Claude API and exports to training format

set -e

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BASE_DIR="$PROJECT_ROOT"
OUTPUT_FILE="$BASE_DIR/data/cengbot_qa_augmented.jsonl"   # --output FILE
LOCK_FILE="${BASE_DIR}/.augmentation_lock"
OWN_LOCK=0

# Remove the lock on exit - but only if it is OUR lock (never another run's)
cleanup() {
    if (( OWN_LOCK == 1 )) && [ -f "$LOCK_FILE" ] && [ "$(cat "$LOCK_FILE" 2>/dev/null)" = "$$" ]; then
        rm -f "$LOCK_FILE"
        print_status "Lock file cleaned up"
    fi
}
trap cleanup EXIT

# Function to check if process is already running
check_lock_file() {
    if [ -f "$LOCK_FILE" ]; then
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || true)
        if [[ "$LOCK_PID" =~ ^[0-9]+$ ]] && ps -p "$LOCK_PID" > /dev/null 2>&1; then
            print_error "Another augmentation process is already running (PID: $LOCK_PID)"
            print_status "If you're sure no other process is running, remove: $LOCK_FILE"
            exit 1
        else
            print_warning "Stale lock file found, removing..."
            rm -f "$LOCK_FILE"
        fi
    fi
}

# Function to create lock file
create_lock_file() {
    if ! ( set -o noclobber; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
        print_error "Another augmentation process just started (lock: $LOCK_FILE)"
        exit 1
    fi
    OWN_LOCK=1
    print_status "Lock file created (PID: $$)"
}

# Function to check system prerequisites
check_prerequisites() {
    print_header "Checking System Prerequisites"
    
    # Check if we're in the right directory
    if [ ! -f "$BASE_DIR/src/data_augmentation.py" ]; then
        print_error "Data augmentation script not found at: $BASE_DIR/src/data_augmentation.py"
        exit 1
    fi
    
    # Check the project interpreter
    require_python
    
    # Check if anthropic package is installed (not needed for --dry-run)
    if (( DRY_RUN == 0 )) && ! "$PYTHON" -c "import anthropic" 2>/dev/null; then
        print_warning "The 'anthropic' package is not installed in $PYTHON"
        if confirm "Install it now ($PYTHON -m pip install anthropic)?"; then
            "$PYTHON" -m pip install anthropic
        else
            print_error "Install it with: $PYTHON -m pip install anthropic"
            exit 1
        fi
    fi
    
    print_success "All prerequisites satisfied"
}

# Function to validate database has training data
validate_database() {
    print_header "Validating Database Content"
    
    # Check if there's usable training data (active, not rejected)
    # Through the ORM, so DATABASE_URL is honoured
    result=$(cd "$BASE_DIR" && "$PYTHON" -c '
import sys
sys.path.insert(0, "src")
from sqlalchemy import or_
from database_models import SessionLocal, TrainingData
db = SessionLocal()
try:
    # same filter as the export: active and not rejected (NULL = not rejected)
    print(db.query(TrainingData).filter(
        TrainingData.is_active == True,  # noqa: E712
        or_(TrainingData.review_status.is_(None), TrainingData.review_status != "rejected"),
    ).count())
except Exception:
    print(0)
finally:
    db.close()
' 2>/dev/null | tail -n 1) || result=0
    [[ "$result" =~ ^[0-9]+$ ]] || result=0
    
    if [ "$result" -eq 0 ]; then
        print_error "No usable training data (active, not rejected) found in database"
        print_status "Please approve some training data in the admin panel first"
        exit 1
    fi
    
    print_success "Found $result usable training records (active, not rejected) in database"
}

# Function to show cost warning
show_cost_warning() {
    print_header "Cost and Usage Warning"
    
    echo -e "${YELLOW}IMPORTANT NOTICES:${NC}"
    echo "• This script uses Anthropic's Claude API which incurs costs"
    echo "• Each question generates ~15 variations using Claude"
    echo "• API costs depend on input/output token usage"
    echo "• API key: taken from ANTHROPIC_API_KEY / CLAUDE_API_KEY (environment or .env),"
    echo "  otherwise asked (hidden input); it is never stored or printed"
    echo "• Process uses checkpoints - safe to interrupt"
    echo
    echo -e "${YELLOW}SECURITY:${NC}"
    echo "• No credentials are saved to files or logs"
    echo
    echo -e "${YELLOW}PROCESS:${NC}"
    echo "• Loads approved training data from database"
    echo "• Generates variations using Claude API"
    echo "• Saves to $OUTPUT_FILE"
    echo "• Runs database export script afterward"
    echo
}

# Function to run augmentation
run_augmentation() {
    print_header "Running Data Augmentation"
    
    # Change to project directory
    cd "$BASE_DIR"
    
    # Activate virtual environment and run augmentation
    print_status "Starting data augmentation process..."
    
    # Run the augmentation script
    # The user already confirmed in this script -> --yes (otherwise they are asked twice)
    local yes=()
    (( DRY_RUN == 1 )) || yes=(--yes)
    if "$PYTHON" src/data_augmentation.py "${yes[@]+"${yes[@]}"}" "${AUG_ARGS[@]+"${AUG_ARGS[@]}"}"; then
        print_success "Data augmentation completed successfully"
    else
        print_error "Data augmentation failed"
        exit 1
    fi
}

# Function to export training data
export_training_data() {
    print_header "Exporting Training Data"
    
    # Check if export script exists
    if [ ! -f "$BASE_DIR/scripts/export_training_data.sh" ]; then
        print_error "Export script not found at: $BASE_DIR/scripts/export_training_data.sh"
        exit 1
    fi
    
    # Run export script
    print_status "Running database export script..."
    
    if "$BASE_DIR/scripts/export_training_data.sh" --output "$OUTPUT_FILE"; then
        print_success "Training data export completed"
    else
        print_error "Training data export failed"
        exit 1
    fi
}

# Function to show final summary
show_summary() {
    print_header "Augmentation Process Summary"
    
    # Check output file
    if [ -f "$OUTPUT_FILE" ]; then
        TOTAL_RECORDS=$(wc -l < "$OUTPUT_FILE")
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        
        print_success "Augmentation completed successfully!"
        echo "📊 Results:"
        echo "   • Total records: $TOTAL_RECORDS"
        echo "   • File size: $FILE_SIZE"
        echo "   • Output file: $OUTPUT_FILE"
        echo
        echo "🔄 Next steps:"
        echo "   • Review the augmented data quality"
        echo "   • Check the data: ./scripts/train_model.sh --dry-run"
        echo "   • Run training:   ./scripts/train_model.sh   (uses data/merged_qa_data.jsonl + data/cengbot_qa_augmented.jsonl)"
        echo "   • Activate:       ./scripts/switch_model.sh switch <new-model>   (docs/MODEL_SWAP.md)"
        echo "   • Monitor training progress"
        echo
    else
        print_error "Output file not found - augmentation may have failed"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "CengBot Training Data Augmentation Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -n, --dry-run      Show what would be augmented (no API calls, no writes)"
    echo "  --max-items N      Process at most N database rows (cost cap)"
    echo "  --model ID         Claude model id (default: env CLAUDE_MODEL or the built-in default)"
    echo "  --reset            Ignore checkpoint/existing output and re-augment everything"
    echo "  --output FILE      Output JSONL (default: data/cengbot_qa_augmented.jsonl); the"
    echo "                     database export afterwards merges into the same file"
    echo "  -y, --yes          Do not ask for confirmation (spends API credits!)"
    echo "  -h, --help         Show this help message"
    echo
    echo "Description:"
    echo "  This script augments training data using Anthropic's Claude API."
    echo "  It loads approved training data from the database, generates"
    echo "  variations using AI, and exports to training format."
    echo
    echo "Prerequisites:"
    echo "  • Anthropic API key (will be requested)"
    echo "  • Approved training data in database"
    echo "  • Python virtual environment with dependencies"
    echo
    echo "Security:"
    echo "  • API key is NOT stored anywhere"
    echo "  • Requested fresh each time"
    echo "  • Cleaned from memory after use"
    echo
    echo "Process:"
    echo "  1. Check system prerequisites"
    echo "  2. Validate database content"
    echo "  3. Show cost warning"
    echo "  4. Request API key from user"
    echo "  5. Run data augmentation"
    echo "  6. Export training data"
    echo "  7. Show summary"
    echo
}

# Parse command line arguments
# shellcheck disable=SC2034  # accepted for compatibility, currently no extra output
VERBOSE=false
DRY_RUN=0
ASSUME_YES="${ASSUME_YES:-0}"
AUG_ARGS=()   # forwarded to src/data_augmentation.py

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            # shellcheck disable=SC2034
            VERBOSE=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=1; AUG_ARGS+=(--dry-run); shift ;;
        -y|--yes)
            ASSUME_YES=1; shift ;;
        --reset)
            AUG_ARGS+=(--reset); shift ;;
        --max-items|--model)
            [[ $# -ge 2 ]] || { print_error "$1 needs a value"; exit 1; }
            AUG_ARGS+=("$1" "$2"); shift 2 ;;
        --max-items=*|--model=*)
            AUG_ARGS+=("$1"); shift ;;
        --output)
            [[ $# -ge 2 && -n "$2" ]] || { print_error "$1 needs a file"; exit 1; }
            OUTPUT_FILE="$2"; shift 2 ;;
        --output=*)
            OUTPUT_FILE="${1#--output=}"; shift ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Relative --output paths are relative to the caller's directory (the steps below run in the project root)
[[ "$OUTPUT_FILE" == /* ]] || OUTPUT_FILE="$PWD/$OUTPUT_FILE"
AUG_ARGS+=(--output "$OUTPUT_FILE")

# Main execution
main() {
    print_header "CengBot Training Data Augmentation"
    
    # Check for running processes
    check_lock_file
    
    # Create lock file
    create_lock_file
    
    # Check prerequisites
    check_prerequisites
    
    # Validate database
    validate_database
    
    if (( DRY_RUN == 1 )); then
        run_augmentation
        print_success "Dry run finished - no API calls, nothing written"
        return 0
    fi

    # Show cost warning
    show_cost_warning
    
    # Confirm with user
    if [[ "$ASSUME_YES" != "1" ]]; then
        echo -e "${YELLOW}❓ Do you want to continue with data augmentation?${NC}"
        read -r -p "Enter 'YES' to continue: " answer || answer=""
        if [ "$answer" != "YES" ]; then
            print_error "Process cancelled by user"
            exit 0
        fi
    fi
    
    # Run augmentation
    run_augmentation
    
    # Export training data
    export_training_data
    
    # Show summary
    show_summary
    
    print_success "🎉 All processes completed successfully!"
}

# Run main function
main "$@"
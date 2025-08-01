#!/bin/bash

# CengBot Training Data Augmentation Script
# Augments training data using Anthropic Claude API and exports to training format

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_DIR="/home/ceng/cu_ceng_bot"
PYTHON_VENV="${BASE_DIR}/.venv"
LOCK_FILE="${BASE_DIR}/.augmentation_lock"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

# Function to cleanup on exit
cleanup() {
    if [ -f "$LOCK_FILE" ]; then
        rm -f "$LOCK_FILE"
        print_status "Lock file cleaned up"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Function to check if process is already running
check_lock_file() {
    if [ -f "$LOCK_FILE" ]; then
        LOCK_PID=$(cat "$LOCK_FILE")
        if ps -p "$LOCK_PID" > /dev/null 2>&1; then
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
    echo $$ > "$LOCK_FILE"
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
    
    # Check if database exists
    if [ ! -f "$BASE_DIR/university_bot.db" ]; then
        print_error "Database not found at: $BASE_DIR/university_bot.db"
        print_status "Please run the system first to create the database"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "$PYTHON_VENV" ]; then
        print_error "Python virtual environment not found at: $PYTHON_VENV"
        print_status "Please run: python3 -m venv $PYTHON_VENV"
        exit 1
    fi
    
    # Check if anthropic package is installed
    if ! "$PYTHON_VENV/bin/pip" list | grep -q anthropic; then
        print_warning "Anthropic package not found in virtual environment"
        print_status "Installing anthropic package..."
        "$PYTHON_VENV/bin/pip" install anthropic
    fi
    
    print_success "All prerequisites satisfied"
}

# Function to validate database has training data
validate_database() {
    print_header "Validating Database Content"
    
    # Check if there's approved training data
    result=$(python3 -c "
import sys
sys.path.append('$BASE_DIR/src')
from database_models import SessionLocal, TrainingData

db = SessionLocal()
try:
    count = db.query(TrainingData).filter(TrainingData.is_active == True).count()
    print(count)
except Exception as e:
    print(0)
finally:
    db.close()
")
    
    if [ "$result" -eq 0 ]; then
        print_error "No approved training data found in database"
        print_status "Please approve some training data in the admin panel first"
        exit 1
    fi
    
    print_success "Found $result approved training records in database"
}

# Function to show cost warning
show_cost_warning() {
    print_header "Cost and Usage Warning"
    
    echo -e "${YELLOW}IMPORTANT NOTICES:${NC}"
    echo "• This script uses Anthropic's Claude API which incurs costs"
    echo "• Each question generates ~15 variations using Claude"
    echo "• API costs depend on input/output token usage"
    echo "• Your API key will NOT be stored anywhere"
    echo "• API key is only used for this session"
    echo "• Process uses checkpoints - safe to interrupt"
    echo
    echo -e "${YELLOW}SECURITY:${NC}"
    echo "• API key is requested each time (not stored)"
    echo "• No credentials are saved to files"
    echo "• Process cleans up API key from memory"
    echo
    echo -e "${YELLOW}PROCESS:${NC}"
    echo "• Loads approved training data from database"
    echo "• Generates variations using Claude API"
    echo "• Saves to data/cengbot_qa_augmented.jsonl"
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
    if "$PYTHON_VENV/bin/python" src/data_augmentation.py; then
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
    
    if "$BASE_DIR/scripts/export_training_data.sh"; then
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
    if [ -f "$BASE_DIR/data/cengbot_qa_augmented.jsonl" ]; then
        TOTAL_RECORDS=$(wc -l < "$BASE_DIR/data/cengbot_qa_augmented.jsonl")
        FILE_SIZE=$(du -h "$BASE_DIR/data/cengbot_qa_augmented.jsonl" | cut -f1)
        
        print_success "Augmentation completed successfully!"
        echo "📊 Results:"
        echo "   • Total records: $TOTAL_RECORDS"
        echo "   • File size: $FILE_SIZE"
        echo "   • Output file: data/cengbot_qa_augmented.jsonl"
        echo
        echo "🔄 Next steps:"
        echo "   • Review the augmented data quality"
        echo "   • Run training: ./scripts/train_model.sh"
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
    echo "  -h, --help     Show this help message"
    echo "  -v, --verbose  Enable verbose output"
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
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

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
    
    # Show cost warning
    show_cost_warning
    
    # Confirm with user
    echo -e "${YELLOW}❓ Do you want to continue with data augmentation?${NC}"
    read -p "Enter 'YES' to continue: " confirm
    
    if [ "$confirm" != "YES" ]; then
        print_error "Process cancelled by user"
        exit 0
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
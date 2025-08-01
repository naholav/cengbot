#!/bin/bash

# CengBot Database to Training Data Export Script
# Exports approved training data from database to JSONL format

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BASE_DIR="/home/ceng/cu_ceng_bot"
SRC_DIR="${BASE_DIR}/src"
DATA_DIR="${BASE_DIR}/data"
EXPORT_SCRIPT="${SRC_DIR}/database_to_training.py"

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

# Function to check database
check_database() {
    print_status "Checking database..."
    
    if [ ! -f "${BASE_DIR}/university_bot.db" ]; then
        print_error "Database not found: ${BASE_DIR}/university_bot.db"
        return 1
    fi
    
    # Check if database has training data
    training_count=$(python3 -c "
import sqlite3
conn = sqlite3.connect('${BASE_DIR}/university_bot.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM training_data WHERE is_active = 1')
count = cursor.fetchone()[0]
conn.close()
print(count)
" 2>/dev/null || echo "0")
    
    if [ "$training_count" -eq 0 ]; then
        print_warning "No active training data found in database"
        return 1
    fi
    
    print_success "Database contains $training_count active training records"
    return 0
}

# Function to export training data
export_training_data() {
    print_header "Exporting Training Data from Database"
    
    # Change to base directory
    cd "$BASE_DIR"
    
    # Check if export script exists
    if [ ! -f "$EXPORT_SCRIPT" ]; then
        print_error "Export script not found: $EXPORT_SCRIPT"
        return 1
    fi
    
    # Run the export
    print_status "Running database export..."
    
    if python3 "$EXPORT_SCRIPT" "$@"; then
        print_success "Database export completed successfully!"
        
        # Show file info
        if [ -f "${DATA_DIR}/cengbot_qa_augmented.jsonl" ]; then
            line_count=$(wc -l < "${DATA_DIR}/cengbot_qa_augmented.jsonl")
            file_size=$(du -h "${DATA_DIR}/cengbot_qa_augmented.jsonl" | cut -f1)
            print_status "Training file: ${DATA_DIR}/cengbot_qa_augmented.jsonl"
            print_status "  Lines: $line_count"
            print_status "  Size: $file_size"
            
            # Show sample data
            print_status "Sample training data:"
            head -3 "${DATA_DIR}/cengbot_qa_augmented.jsonl" | python3 -c "
import sys
import json
for i, line in enumerate(sys.stdin, 1):
    try:
        data = json.loads(line.strip())
        print(f'  {i}. ID: {data[\"id\"]}, Language: {data[\"language\"]}')
        print(f'     Q: {data[\"question\"][:50]}...')
        print(f'     A: {data[\"answer\"][:50]}...')
        print()
    except:
        pass
"
        fi
    else
        print_error "Database export failed!"
        return 1
    fi
}

# Function to show help
show_help() {
    echo "CengBot Database to Training Data Export"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --include-inactive    Include inactive training records"
    echo "  --no-backup          Skip creating backup file"
    echo "  --output FILE        Specify output file path"
    echo "  --help               Show this help message"
    echo
    echo "Examples:"
    echo "  $0                                    # Export active training data"
    echo "  $0 --include-inactive                # Export all training data"
    echo "  $0 --no-backup                       # Export without backup"
    echo "  $0 --output /path/to/custom.jsonl    # Export to custom file"
}

# Function to validate environment
validate_environment() {
    print_status "Validating environment..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found"
        return 1
    fi
    
    # Check required Python modules
    python3 -c "
import sys
try:
    import sqlite3
    import json
    from pathlib import Path
    print('✓ Required modules available')
except ImportError as e:
    print(f'✗ Missing module: {e}')
    sys.exit(1)
" || return 1
    
    # Check directories
    if [ ! -d "$BASE_DIR" ]; then
        print_error "Base directory not found: $BASE_DIR"
        return 1
    fi
    
    if [ ! -d "$SRC_DIR" ]; then
        print_error "Source directory not found: $SRC_DIR"
        return 1
    fi
    
    # Create data directory if needed
    mkdir -p "$DATA_DIR"
    
    print_success "Environment validation passed"
    return 0
}

# Main function
main() {
    print_header "CengBot Database to Training Data Export"
    
    # Handle help
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_help
        return 0
    fi
    
    # Validate environment
    if ! validate_environment; then
        return 1
    fi
    
    # Check database
    if ! check_database; then
        return 1
    fi
    
    # Export training data
    if ! export_training_data "$@"; then
        return 1
    fi
    
    print_success "Export process completed successfully!"
    return 0
}

# Run main function
main "$@"
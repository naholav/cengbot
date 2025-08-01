#!/bin/bash
#
# Automated Database Export Script for CengBot
# ===========================================
#
# This script automatically exports the CengBot database to Excel format.
# Includes logging and error handling.
#
# Usage:
#   ./export_database.sh [OPTIONS]
#
# Options:
#   -h, --help    Show this help message
#   -v, --verbose Enable verbose logging
#   -q, --quiet   Suppress output (except errors)
#
# Author: naholav
#

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/excel_export.log"
PYTHON_SCRIPT="$PROJECT_ROOT/src/export_to_excel.py"
VENV_DIR="$PROJECT_ROOT/.venv"
CURRENT_PYTHON="$(which python)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
VERBOSE=false
QUIET=false

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${color}${message}${NC}"
    fi
}

# Function to log messages
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create log directory if it doesn't exist
    mkdir -p "$LOG_DIR"
    
    # Write to log file
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    
    # Print to console based on level and options
    case $level in
        "ERROR")
            print_color "$RED" "❌ $message"
            ;;
        "WARN")
            print_color "$YELLOW" "⚠️  $message"
            ;;
        "INFO")
            if [[ "$VERBOSE" == "true" ]]; then
                print_color "$BLUE" "ℹ️  $message"
            fi
            ;;
        "SUCCESS")
            print_color "$GREEN" "✅ $message"
            ;;
        *)
            if [[ "$QUIET" != "true" ]]; then
                echo "$message"
            fi
            ;;
    esac
}

# Function to show help
show_help() {
    cat << EOF
CengBot Database Export Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help      Show this help message and exit
    -v, --verbose   Enable verbose logging
    -q, --quiet     Suppress output (except errors)

DESCRIPTION:
    This script exports the CengBot SQLite database to Excel format.
    The export includes all tables with timestamp in filename.

EXAMPLES:
    $0                  # Normal export
    $0 -v               # Verbose export
    $0 -q               # Quiet export

FILES:
    Input:  $PROJECT_ROOT/university_bot.db
    Output: $PROJECT_ROOT/excel/cengbot_database_export_YYYYMMDD_HHMMSS.xlsx
    Log:    $LOG_FILE

EOF
}

# Parse command line options
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
        -q|--quiet)
            QUIET=true
            shift
            ;;
        *)
            log_message "ERROR" "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check requirements
check_requirements() {
    log_message "INFO" "Checking requirements..."
    
    # Check if database file exists
    if [[ ! -f "$PROJECT_ROOT/university_bot.db" ]]; then
        log_message "ERROR" "Database file not found: $PROJECT_ROOT/university_bot.db"
        return 1
    fi
    
    # Check if Python script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_message "ERROR" "Python export script not found: $PYTHON_SCRIPT"
        return 1
    fi
    
    # Check if required Python packages are installed in current environment
    if ! python -c "import pandas, openpyxl, sqlalchemy" 2>/dev/null; then
        log_message "ERROR" "Required Python packages not found in current environment"
        log_message "ERROR" "Please install: pip install pandas openpyxl sqlalchemy"
        log_message "ERROR" "Current Python: $CURRENT_PYTHON"
        return 1
    fi
    
    log_message "SUCCESS" "All requirements satisfied"
    return 0
}

# Function to backup previous export if exists
backup_previous_export() {
    local latest_export=$(ls -t "$PROJECT_ROOT/excel"/cengbot_database_export_*.xlsx 2>/dev/null | head -n 1)
    if [[ -n "$latest_export" ]]; then
        local backup_name="${latest_export}.backup"
        cp "$latest_export" "$backup_name" 2>/dev/null
        log_message "INFO" "Previous export backed up as: $(basename "$backup_name")"
    fi
}

# Function to run the export
run_export() {
    log_message "INFO" "Starting database export..."
    
    # Change to project root directory
    cd "$PROJECT_ROOT" || {
        log_message "ERROR" "Failed to change to project root directory"
        return 1
    }
    
    # Use current Python environment (already activated)
    
    # Run the Python export script
    if [[ "$VERBOSE" == "true" ]]; then
        python "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
        local exit_code=${PIPESTATUS[0]}
    else
        python "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1
        local exit_code=$?
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        log_message "SUCCESS" "Database export completed successfully"
        
        # Show the latest created file
        local latest_file=$(ls -t "$PROJECT_ROOT/excel"/cengbot_database_export_*.xlsx 2>/dev/null | head -n 1)
        if [[ -n "$latest_file" ]]; then
            local file_size=$(stat -f%z "$latest_file" 2>/dev/null || stat -c%s "$latest_file" 2>/dev/null)
            local file_size_mb=$(echo "scale=2; $file_size/1024/1024" | bc 2>/dev/null || echo "unknown")
            log_message "SUCCESS" "Export file: $(basename "$latest_file") (${file_size_mb} MB)"
            log_message "SUCCESS" "Location: $latest_file"
        fi
        
        return 0
    else
        log_message "ERROR" "Database export failed with exit code: $exit_code"
        return 1
    fi
}

# Function to cleanup old exports (keep last 10)
cleanup_old_exports() {
    log_message "INFO" "Cleaning up old export files..."
    
    cd "$PROJECT_ROOT/excel" || return
    
    # Keep only the 10 most recent export files
    local old_files=$(ls -t cengbot_database_export_*.xlsx 2>/dev/null | tail -n +11)
    
    if [[ -n "$old_files" ]]; then
        echo "$old_files" | while read -r file; do
            rm -f "$file"
            log_message "INFO" "Removed old export: $file"
        done
        log_message "SUCCESS" "Cleanup completed"
    else
        log_message "INFO" "No old files to cleanup"
    fi
}

# Function to show export statistics
show_statistics() {
    if [[ "$QUIET" == "true" ]]; then
        return
    fi
    
    echo ""
    print_color "$BLUE" "📊 Export Statistics:"
    print_color "$BLUE" "==================="
    
    # Count export files
    local export_count=$(ls -1 "$PROJECT_ROOT/excel"/cengbot_database_export_*.xlsx 2>/dev/null | wc -l)
    print_color "$BLUE" "Total exports: $export_count"
    
    # Show database size
    if [[ -f "$PROJECT_ROOT/university_bot.db" ]]; then
        local db_size=$(stat -f%z "$PROJECT_ROOT/university_bot.db" 2>/dev/null || stat -c%s "$PROJECT_ROOT/university_bot.db" 2>/dev/null)
        local db_size_mb=$(echo "scale=2; $db_size/1024/1024" | bc 2>/dev/null || echo "unknown")
        print_color "$BLUE" "Database size: ${db_size_mb} MB"
    fi
    
    # Show latest export
    local latest_file=$(ls -t "$PROJECT_ROOT/excel"/cengbot_database_export_*.xlsx 2>/dev/null | head -n 1)
    if [[ -n "$latest_file" ]]; then
        local export_date=$(stat -f%Sm -t"%Y-%m-%d %H:%M" "$latest_file" 2>/dev/null || stat -c%y "$latest_file" 2>/dev/null | cut -d' ' -f1,2 | cut -d':' -f1,2)
        print_color "$BLUE" "Latest export: $export_date"
    fi
    
    echo ""
}

# Main execution
main() {
    # Print header
    if [[ "$QUIET" != "true" ]]; then
        echo ""
        print_color "$GREEN" "🚀 CengBot Database Export Script"
        print_color "$GREEN" "=================================="
        echo ""
    fi
    
    log_message "INFO" "Starting export process..."
    
    # Check all requirements
    if ! check_requirements; then
        log_message "ERROR" "Requirements check failed"
        exit 1
    fi
    
    # Backup previous export
    backup_previous_export
    
    # Run the export
    if ! run_export; then
        log_message "ERROR" "Export process failed"
        exit 1
    fi
    
    # Cleanup old files
    cleanup_old_exports
    
    # Show statistics
    show_statistics
    
    log_message "SUCCESS" "Export process completed successfully!"
    
    if [[ "$QUIET" != "true" ]]; then
        echo ""
        print_color "$GREEN" "🎉 All done! Check the excel/ folder for your export file."
        echo ""
    fi
}

# Run main function
main "$@"
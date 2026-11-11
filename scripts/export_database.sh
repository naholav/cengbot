#!/usr/bin/env bash
#
# Automated Database Export Script for CengBot
# ===========================================
#
# Exports the CengBot database to Excel with src/export_to_excel.py (project
# venv). Default: the raw_data table -> excel/cengbot_database_export_<ts>.xlsx.
# Keeps the 10 newest default exports in excel/ (KEEP_EXPORTS=N to change).
#
# Usage:
#   ./export_database.sh [OPTIONS] [export_to_excel.py options]
#
# Options:
#   -h, --help    Show this help message
#   -v, --verbose Show the exporter output (always written to logs/excel_export.log)
#   -q, --quiet   Suppress output (except errors)
#   Any other option is passed to src/export_to_excel.py, e.g.
#   --all-tables, --tables raw_data,training_data, --output-dir DIR, --output FILE
#
# Author: naholav
#

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

LOG_FILE="$LOG_DIR/excel_export.log"
PYTHON_SCRIPT="$PROJECT_ROOT/src/export_to_excel.py"
EXCEL_DIR="$PROJECT_ROOT/excel"
KEEP_EXPORTS="${KEEP_EXPORTS:-10}"

VERBOSE=false
QUIET=false
FORWARD=()
CUSTOM_OUTPUT=false

print_color() {
    if [[ "$QUIET" != "true" ]]; then
        echo -e "${1}${2}${NC}"
    fi
}

log_message() {
    local level=$1 message=$2 timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    mkdir -p "$LOG_DIR"
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    case $level in
        ERROR)   echo -e "${RED}❌ $message${NC}" >&2 ;;
        WARN)    print_color "$YELLOW" "⚠️  $message" ;;
        INFO)    [[ "$VERBOSE" == "true" ]] && print_color "$BLUE" "ℹ️  $message" ;;
        SUCCESS) print_color "$GREEN" "✅ $message" ;;
        *)       [[ "$QUIET" != "true" ]] && echo "$message" ;;
    esac
    return 0
}

show_help() {
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    echo "Exporter options:"
    if [[ -x "$PYTHON" ]]; then
        "$PYTHON" "$PYTHON_SCRIPT" --help 2>/dev/null | sed 's/^/    /' || true
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)    show_help; exit 0 ;;
        -v|--verbose) VERBOSE=true ;;
        -q|--quiet)   QUIET=true ;;
        --output|--output=*|--output-dir|--output-dir=*)
            CUSTOM_OUTPUT=true; FORWARD+=("$1") ;;
        *)            FORWARD+=("$1") ;;
    esac
    shift
done

check_requirements() {
    log_message "INFO" "Checking requirements..."
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_message "ERROR" "Python export script not found: $PYTHON_SCRIPT"
        return 1
    fi
    if [[ ! -x "$PYTHON" ]]; then
        log_message "ERROR" "Project Python not found: $PYTHON (create .venv or set CENGBOT_PYTHON)"
        return 1
    fi
    if ! "$PYTHON" -c "import pandas, openpyxl, sqlalchemy" 2>/dev/null; then
        log_message "ERROR" "pandas / openpyxl / sqlalchemy missing in $PYTHON"
        log_message "ERROR" "Install them: $PYTHON -m pip install pandas openpyxl sqlalchemy"
        return 1
    fi
    # The database location (DATABASE_URL) is resolved by the exporter itself;
    # it exits with a clear error if the database is missing.
    log_message "SUCCESS" "All requirements satisfied"
    return 0
}

latest_export() {
    find "$EXCEL_DIR" -maxdepth 1 -type f -name 'cengbot_database_export_*.xlsx' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -n 1 | cut -d' ' -f2-
}

run_export() {
    local exit_code
    log_message "INFO" "Starting database export..."
    cd "$PROJECT_ROOT" || { log_message "ERROR" "Failed to change to $PROJECT_ROOT"; return 1; }
    if [[ "$VERBOSE" == "true" ]]; then
        "$PYTHON" "$PYTHON_SCRIPT" "${FORWARD[@]+"${FORWARD[@]}"}" 2>&1 | tee -a "$LOG_FILE"
        exit_code=${PIPESTATUS[0]}
    else
        "$PYTHON" "$PYTHON_SCRIPT" "${FORWARD[@]+"${FORWARD[@]}"}" >> "$LOG_FILE" 2>&1
        exit_code=$?
    fi
    if [[ $exit_code -ne 0 ]]; then
        log_message "ERROR" "Database export failed with exit code $exit_code (details: $LOG_FILE)"
        return 1
    fi
    log_message "SUCCESS" "Database export completed successfully"
    if [[ "$CUSTOM_OUTPUT" != "true" ]]; then
        local latest
        latest=$(latest_export)
        if [[ -n "$latest" ]]; then
            log_message "SUCCESS" "Export file: $(basename "$latest") ($(human_size "$(stat -c %s -- "$latest")"))"
            log_message "SUCCESS" "Location: $latest"
        fi
    fi
    return 0
}

# Keep only the newest KEEP_EXPORTS default exports (custom --output paths are never touched)
cleanup_old_exports() {
    local f
    [[ "$CUSTOM_OUTPUT" == "true" || ! -d "$EXCEL_DIR" ]] && return 0
    [[ "$KEEP_EXPORTS" =~ ^[1-9][0-9]*$ ]] || { log_message "WARN" "KEEP_EXPORTS must be a positive number"; return 0; }
    while IFS= read -r f; do
        [[ -n "$f" && "$f" == "$EXCEL_DIR"/cengbot_database_export_*.xlsx ]] || continue
        rm -f -- "$f"
        log_message "INFO" "Removed old export: $(basename "$f")"
    done < <(find "$EXCEL_DIR" -maxdepth 1 -type f -name 'cengbot_database_export_*.xlsx' -printf '%T@ %p\n' \
             | sort -rn | tail -n +"$(( KEEP_EXPORTS + 1 ))" | cut -d' ' -f2-)
}

main() {
    if [[ "$QUIET" != "true" ]]; then
        echo ""
        print_color "$GREEN" "🚀 CengBot Database Export Script"
        print_color "$GREEN" "=================================="
        echo ""
    fi
    log_message "INFO" "Starting export process..."
    check_requirements || { log_message "ERROR" "Requirements check failed"; exit 1; }
    run_export || { log_message "ERROR" "Export process failed"; exit 1; }
    cleanup_old_exports
    log_message "SUCCESS" "Export process completed successfully!"
    if [[ "$QUIET" != "true" && "$CUSTOM_OUTPUT" != "true" ]]; then
        local count
        count=$(find "$EXCEL_DIR" -maxdepth 1 -type f -name 'cengbot_database_export_*.xlsx' 2>/dev/null | wc -l)
        print_color "$BLUE" "📊 Exports in excel/: $count (keeping the newest $KEEP_EXPORTS)"
    fi
}

main

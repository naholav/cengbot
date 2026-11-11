#!/usr/bin/env bash

# CengBot Training History Viewer
# View and analyze training session logs and metadata (read-only)
# (logs/training_history/v<N>.log + v<N>_info.json written by src/train_model.py)

set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TRAINING_HISTORY_DIR="${LOG_DIR}/training_history"
PY="$PYTHON"
[[ -x "$PY" ]] || PY="$(command -v python3 || echo python3)"

# check_version V -> only "v<number>" is accepted (no paths / quotes)
check_version() {
    if [[ ! "${1:-}" =~ ^v[0-9]+$ ]]; then
        print_error "Invalid version '${1:-}' - expected e.g. v1, v2"
        exit 1
    fi
}

# Function to list all training sessions
list_training_sessions() {
    print_header "Training Sessions"
    
    if [ ! -d "$TRAINING_HISTORY_DIR" ]; then
        print_error "Training history directory not found: $TRAINING_HISTORY_DIR"
        return 1
    fi
    
    # Count sessions
    session_count=$(find "$TRAINING_HISTORY_DIR" -maxdepth 1 -name 'v*.log' | wc -l)
    
    if [ "$session_count" -eq 0 ]; then
        print_warning "No training sessions found."
        return 0
    fi
    
    print_status "Found $session_count training sessions:"
    echo
    
    # List sessions with details
    for log_file in $(find "$TRAINING_HISTORY_DIR" -maxdepth 1 -name 'v*.log' -printf '%f\n' | sort -V); do
        log_file="$TRAINING_HISTORY_DIR/$log_file"
        if [ -f "$log_file" ]; then
            version=$(basename "$log_file" .log)
            info_file="${TRAINING_HISTORY_DIR}/${version}_info.json"
            
            printf "%-8s" "$version:"
            printf "%-12s" "$(wc -l < "$log_file") lines"
            printf "%-20s" "$(stat -c %y "$log_file" | cut -d' ' -f1)"
            
            if [ -f "$info_file" ]; then
                loss=$("$PY" -I -c '
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        print("Loss: %s" % json.load(f).get("final_loss", "N/A"))
except Exception:
    print("Loss: N/A")
' "$info_file" 2>/dev/null)
                printf "%-15s" "$loss"
            else
                printf "%-15s" "No metadata"
            fi
            echo
        fi
    done
}

# Function to view specific training session
view_session() {
    local version="$1"
    
    if [ -z "$version" ]; then
        print_error "Please specify a version (e.g., v1, v2, v3)"
        return 1
    fi
    check_version "$version"
    
    log_file="${TRAINING_HISTORY_DIR}/${version}.log"
    info_file="${TRAINING_HISTORY_DIR}/${version}_info.json"
    
    if [ ! -f "$log_file" ]; then
        print_error "Training log not found: $log_file"
        return 1
    fi
    
    print_header "Training Session: $version"
    
    # Show metadata if available
    if [ -f "$info_file" ]; then
        print_status "Training Configuration:"
        "$PY" -I -c '
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    for label, key in (("Version", "version"), ("Model", "model_version"), ("Final Loss", "final_loss"),
                       ("Total Steps", "total_steps"), ("Dataset Size", "dataset_size"),
                       ("Learning Rate", "learning_rate"), ("Batch Size", "batch_size"),
                       ("Epochs", "num_epochs"), ("Timestamp", "timestamp")):
        print("  %s: %s" % (label, data.get(key, "N/A")))
except Exception as e:
    print("  Error reading metadata: %s" % e)
' "$info_file" 2>/dev/null
        echo
    fi
    
    # Show log statistics
    print_status "Log Statistics:"
    echo "  Total Lines: $(wc -l < "$log_file")"
    echo "  File Size: $(du -h "$log_file" | cut -f1)"
    echo "  Created: $(stat -c %y "$log_file")"
    echo
    
    # Show recent log entries
    print_status "Recent Log Entries (last 10 lines):"
    tail -n 10 -- "$log_file"
}

# Function to compare training sessions
compare_sessions() {
    local version1="$1"
    local version2="$2"
    
    if [ -z "$version1" ] || [ -z "$version2" ]; then
        print_error "Please specify two versions to compare (e.g., v1 v2)"
        return 1
    fi
    check_version "$version1"
    check_version "$version2"
    
    info_file1="${TRAINING_HISTORY_DIR}/${version1}_info.json"
    info_file2="${TRAINING_HISTORY_DIR}/${version2}_info.json"
    
    if [ ! -f "$info_file1" ] || [ ! -f "$info_file2" ]; then
        print_error "Metadata files not found for comparison"
        return 1
    fi
    
    print_header "Comparing $version1 vs $version2"
    
    "$PY" -I -c '
import json, sys
f1, f2, v1, v2 = sys.argv[1:5]
try:
    with open(f1, encoding="utf-8") as f:
        data1 = json.load(f)
    with open(f2, encoding="utf-8") as f:
        data2 = json.load(f)
    print("                    %-12s %s" % (v1, v2))
    for label, key in (("Final Loss", "final_loss"), ("Total Steps", "total_steps"),
                       ("Dataset Size", "dataset_size"), ("Learning Rate", "learning_rate"),
                       ("Batch Size", "batch_size"), ("Epochs", "num_epochs")):
        print("%-19s %-12s %s" % (label + ":", data1.get(key, "N/A"), data2.get(key, "N/A")))
    loss1, loss2 = data1.get("final_loss"), data2.get("final_loss")
    if isinstance(loss1, (int, float)) and isinstance(loss2, (int, float)) and loss1:
        improvement = (loss1 - loss2) / loss1 * 100
        print("\nImprovement: %.2f%% (%s)" % (improvement, "better" if improvement > 0 else "worse"))
except Exception as e:
    print("Error comparing sessions: %s" % e)
' "$info_file1" "$info_file2" "$version1" "$version2" 2>/dev/null
}

# Function to find best model
find_best_model() {
    print_header "Finding Best Model"
    
    [[ -d "$TRAINING_HISTORY_DIR" ]] || { print_error "Training history directory not found: $TRAINING_HISTORY_DIR"; return 1; }
    "$PY" -I -c '
import json, os, sys
history_dir = sys.argv[1]
sessions = []
for name in sorted(os.listdir(history_dir)):
    if not name.endswith("_info.json"):
        continue
    try:
        with open(os.path.join(history_dir, name), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        continue
    loss = data.get("final_loss")
    sessions.append((data.get("version", name[:-10]), loss if isinstance(loss, (int, float)) else None))
ranked = sorted(sessions, key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))
if ranked and ranked[0][1] is not None:
    print("Best performing model: %s with loss %s" % ranked[0])
    print("\nAll sessions ranked by performance:")
    for i, (version, loss) in enumerate(ranked, 1):
        print("%d. %s: %s" % (i, version, loss if loss is not None else "N/A"))
else:
    print("No training sessions with loss data found.")
' "$TRAINING_HISTORY_DIR" 2>/dev/null
}

# Function to show help
show_help() {
    echo "CengBot Training History Viewer"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  list                    List all training sessions"
    echo "  view <version>          View specific training session (e.g., v1, v2)"
    echo "  compare <v1> <v2>       Compare two training sessions"
    echo "  best                    Find best performing model"
    echo "  help                    Show this help message"
    echo
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 view v2"
    echo "  $0 compare v1 v2"
    echo "  $0 best"
}

# Main script logic
case "${1:-list}" in
    "list")
        list_training_sessions
        ;;
    "view")
        view_session "${2:-}"
        ;;
    "compare")
        compare_sessions "${2:-}" "${3:-}"
        ;;
    "best")
        find_best_model
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
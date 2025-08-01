#!/bin/bash

# CengBot Training History Viewer
# View and analyze training session logs and metadata

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_DIR="/home/ceng/cu_ceng_bot"
TRAINING_HISTORY_DIR="${BASE_DIR}/logs/training_history"

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

# Function to list all training sessions
list_training_sessions() {
    print_header "Training Sessions"
    
    if [ ! -d "$TRAINING_HISTORY_DIR" ]; then
        print_error "Training history directory not found: $TRAINING_HISTORY_DIR"
        return 1
    fi
    
    # Count sessions
    session_count=$(ls -1 "$TRAINING_HISTORY_DIR"/v*.log 2>/dev/null | wc -l)
    
    if [ "$session_count" -eq 0 ]; then
        print_warning "No training sessions found."
        return 0
    fi
    
    print_status "Found $session_count training sessions:"
    echo
    
    # List sessions with details
    for log_file in "$TRAINING_HISTORY_DIR"/v*.log; do
        if [ -f "$log_file" ]; then
            version=$(basename "$log_file" .log)
            info_file="${TRAINING_HISTORY_DIR}/${version}_info.json"
            
            printf "%-8s" "$version:"
            printf "%-12s" "$(wc -l < "$log_file") lines"
            printf "%-20s" "$(stat -c %y "$log_file" | cut -d' ' -f1)"
            
            if [ -f "$info_file" ]; then
                loss=$(python3 -c "
import json
try:
    with open('$info_file', 'r') as f:
        data = json.load(f)
        print(f\"Loss: {data.get('final_loss', 'N/A')}\")
except:
    print('Loss: N/A')
" 2>/dev/null)
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
        python3 -c "
import json
try:
    with open('$info_file', 'r') as f:
        data = json.load(f)
        print(f'  Version: {data.get(\"version\", \"N/A\")}')
        print(f'  Final Loss: {data.get(\"final_loss\", \"N/A\")}')
        print(f'  Total Steps: {data.get(\"total_steps\", \"N/A\")}')
        print(f'  Dataset Size: {data.get(\"dataset_size\", \"N/A\")}')
        print(f'  Learning Rate: {data.get(\"learning_rate\", \"N/A\")}')
        print(f'  Batch Size: {data.get(\"batch_size\", \"N/A\")}')
        print(f'  Epochs: {data.get(\"num_epochs\", \"N/A\")}')
        print(f'  Timestamp: {data.get(\"timestamp\", \"N/A\")}')
except Exception as e:
    print(f'  Error reading metadata: {e}')
" 2>/dev/null
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
    tail -10 "$log_file"
}

# Function to compare training sessions
compare_sessions() {
    local version1="$1"
    local version2="$2"
    
    if [ -z "$version1" ] || [ -z "$version2" ]; then
        print_error "Please specify two versions to compare (e.g., v1 v2)"
        return 1
    fi
    
    info_file1="${TRAINING_HISTORY_DIR}/${version1}_info.json"
    info_file2="${TRAINING_HISTORY_DIR}/${version2}_info.json"
    
    if [ ! -f "$info_file1" ] || [ ! -f "$info_file2" ]; then
        print_error "Metadata files not found for comparison"
        return 1
    fi
    
    print_header "Comparing $version1 vs $version2"
    
    python3 -c "
import json
try:
    with open('$info_file1', 'r') as f:
        data1 = json.load(f)
    with open('$info_file2', 'r') as f:
        data2 = json.load(f)
    
    print(f'                    $version1          $version2')
    print(f'Final Loss:         {data1.get(\"final_loss\", \"N/A\"):<12} {data2.get(\"final_loss\", \"N/A\")}')
    print(f'Total Steps:        {data1.get(\"total_steps\", \"N/A\"):<12} {data2.get(\"total_steps\", \"N/A\")}')
    print(f'Dataset Size:       {data1.get(\"dataset_size\", \"N/A\"):<12} {data2.get(\"dataset_size\", \"N/A\")}')
    print(f'Learning Rate:      {data1.get(\"learning_rate\", \"N/A\"):<12} {data2.get(\"learning_rate\", \"N/A\")}')
    print(f'Batch Size:         {data1.get(\"batch_size\", \"N/A\"):<12} {data2.get(\"batch_size\", \"N/A\")}')
    print(f'Epochs:             {data1.get(\"num_epochs\", \"N/A\"):<12} {data2.get(\"num_epochs\", \"N/A\")}')
    
    # Calculate improvement
    loss1 = data1.get('final_loss')
    loss2 = data2.get('final_loss')
    if loss1 and loss2:
        improvement = ((loss1 - loss2) / loss1) * 100
        print(f'\\nImprovement: {improvement:.2f}% ({\"better\" if improvement > 0 else \"worse\"})')
    
except Exception as e:
    print(f'Error comparing sessions: {e}')
" 2>/dev/null
}

# Function to find best model
find_best_model() {
    print_header "Finding Best Model"
    
    python3 -c "
import json
import os
history_dir = '$TRAINING_HISTORY_DIR'
best_loss = float('inf')
best_version = None
sessions = []

for file in os.listdir(history_dir):
    if file.endswith('_info.json'):
        try:
            with open(os.path.join(history_dir, file), 'r') as f:
                data = json.load(f)
                loss = data.get('final_loss')
                if loss and loss < best_loss:
                    best_loss = loss
                    best_version = data.get('version', 'Unknown')
                sessions.append((data.get('version', 'Unknown'), loss))
        except:
            continue

if best_version:
    print(f'Best performing model: {best_version} with loss {best_loss}')
    print(f'\\nAll sessions ranked by performance:')
    sessions.sort(key=lambda x: x[1] if x[1] else float('inf'))
    for i, (version, loss) in enumerate(sessions, 1):
        print(f'{i}. {version}: {loss if loss else \"N/A\"}')
else:
    print('No training sessions with loss data found.')
" 2>/dev/null
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
        view_session "$2"
        ;;
    "compare")
        compare_sessions "$2" "$3"
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
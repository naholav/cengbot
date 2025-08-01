#!/bin/bash

# CengBot Model Version Switcher
# Switch between different model versions (v1, v2, v3, etc.)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

BASE_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"
MODELS_DIR="${BASE_DIR}/models"
ACTIVE_MODEL_LINK="${MODELS_DIR}/active-model"

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

# Function to list available models
list_models() {
    print_header "Available Model Versions"
    
    if [ ! -d "$MODELS_DIR" ]; then
        print_error "Models directory not found: $MODELS_DIR"
        return 1
    fi
    
    # Find all model versions
    model_versions=()
    for dir in "$MODELS_DIR"/final-best-model-v*; do
        if [ -d "$dir" ]; then
            version=$(basename "$dir" | sed 's/final-best-model-v//')
            model_versions+=("$version")
        fi
    done
    
    if [ ${#model_versions[@]} -eq 0 ]; then
        print_warning "No model versions found"
        return 0
    fi
    
    # Sort versions numerically
    IFS=$'\n' sorted_versions=($(sort -n <<<"${model_versions[*]}"))
    unset IFS
    
    # Get current active model
    current_version=""
    if [ -L "$ACTIVE_MODEL_LINK" ]; then
        current_target=$(readlink "$ACTIVE_MODEL_LINK")
        current_version=$(basename "$current_target" | sed 's/final-best-model-v//')
    fi
    
    print_status "Found ${#sorted_versions[@]} model versions:"
    echo
    
    for version in "${sorted_versions[@]}"; do
        model_dir="${MODELS_DIR}/final-best-model-v${version}"
        
        # Check if this is the active model
        if [ "$version" = "$current_version" ]; then
            echo -e "  ${GREEN}v${version}${NC} (ACTIVE)"
        else
            echo -e "  v${version}"
        fi
        
        # Show model info if available
        if [ -f "$model_dir/training_info.json" ]; then
            info=$(python3 -c "
import json
try:
    with open('$model_dir/training_info.json', 'r') as f:
        data = json.load(f)
        print(f'    Loss: {data.get(\"final_loss\", \"N/A\")} | Steps: {data.get(\"total_steps\", \"N/A\")} | Date: {data.get(\"timestamp\", \"N/A\")[:10]}')
except:
    print('    No training info available')
" 2>/dev/null)
            echo -e "    ${info}"
        fi
        echo
    done
}

# Function to switch to a specific model version
switch_model() {
    local target_version="$1"
    
    if [ -z "$target_version" ]; then
        print_error "Please specify a model version (e.g., v2, v3)"
        return 1
    fi
    
    # Remove 'v' prefix if present
    target_version=$(echo "$target_version" | sed 's/^v//')
    
    local target_dir="${MODELS_DIR}/final-best-model-v${target_version}"
    
    if [ ! -d "$target_dir" ]; then
        print_error "Model version v${target_version} not found: $target_dir"
        print_status "Available versions:"
        list_models
        return 1
    fi
    
    print_header "Switching to Model v${target_version}"
    
    # Check if model has required files - support both v1 style (with method1) and v1.1 style (direct)
    if [ ! -d "$target_dir/method1" ] && [ ! -f "$target_dir/adapter_model.safetensors" ]; then
        print_error "Model v${target_version} appears to be incomplete (missing method1 directory or adapter files)"
        return 1
    fi
    
    # Remove existing active model link
    if [ -L "$ACTIVE_MODEL_LINK" ]; then
        rm "$ACTIVE_MODEL_LINK"
        print_status "Removed existing active model link"
    fi
    
    # Create new symbolic link
    ln -s "final-best-model-v${target_version}" "$ACTIVE_MODEL_LINK"
    
    # Verify the link was created successfully
    if [ -L "$ACTIVE_MODEL_LINK" ]; then
        print_success "Successfully switched to model v${target_version}"
        
        # Show model info
        if [ -f "$target_dir/training_info.json" ]; then
            print_status "Model Information:"
            python3 -c "
import json
try:
    with open('$target_dir/training_info.json', 'r') as f:
        data = json.load(f)
        print(f'  Version: {data.get(\"model_version\", \"N/A\")}')
        print(f'  Final Loss: {data.get(\"final_loss\", \"N/A\")}')
        print(f'  Total Steps: {data.get(\"total_steps\", \"N/A\")}')
        print(f'  Dataset Size: {data.get(\"dataset_size\", \"N/A\")}')
        print(f'  Training Date: {data.get(\"timestamp\", \"N/A\")[:10]}')
except:
    print('  No training info available')
" 2>/dev/null
        fi
    else
        print_error "Failed to create symbolic link"
        return 1
    fi
}

# Function to get current active model
get_current_model() {
    if [ -L "$ACTIVE_MODEL_LINK" ]; then
        current_target=$(readlink "$ACTIVE_MODEL_LINK")
        current_version=$(basename "$current_target" | sed 's/final-best-model-v//')
        echo "v${current_version}"
    else
        echo "none"
    fi
}

# Function to show current model info
show_current() {
    print_header "Current Active Model"
    
    current_model=$(get_current_model)
    
    if [ "$current_model" = "none" ]; then
        print_warning "No active model set"
        print_status "Use 'switch_model.sh v1' to set an active model"
        return 0
    fi
    
    print_success "Active model: $current_model"
    
    # Show model path
    if [ -L "$ACTIVE_MODEL_LINK" ]; then
        actual_path=$(readlink -f "$ACTIVE_MODEL_LINK")
        print_status "Model path: $actual_path"
        
        # Show model info
        info_file="${actual_path}/training_info.json"
        if [ -f "$info_file" ]; then
            print_status "Model Information:"
            python3 -c "
import json
try:
    with open('$info_file', 'r') as f:
        data = json.load(f)
        print(f'  Version: {data.get(\"model_version\", \"N/A\")}')
        print(f'  Final Loss: {data.get(\"final_loss\", \"N/A\")}')
        print(f'  Total Steps: {data.get(\"total_steps\", \"N/A\")}')
        print(f'  Dataset Size: {data.get(\"dataset_size\", \"N/A\")}')
        print(f'  Training Date: {data.get(\"timestamp\", \"N/A\")[:10]}')
except:
    print('  No training info available')
" 2>/dev/null
        fi
    fi
}

# Function to auto-switch to next version
auto_switch() {
    current_model=$(get_current_model)
    
    if [ "$current_model" = "none" ]; then
        print_status "No active model, switching to v1"
        switch_model "v1"
        return
    fi
    
    current_version=$(echo "$current_model" | sed 's/v//')
    next_version=$((current_version + 1))
    
    if [ -d "${MODELS_DIR}/final-best-model-v${next_version}" ]; then
        print_status "Auto-switching from v${current_version} to v${next_version}"
        switch_model "v${next_version}"
    else
        print_warning "Next version v${next_version} not found"
        print_status "Current model v${current_version} remains active"
    fi
}

# Function to show help
show_help() {
    echo "CengBot Model Version Switcher"
    echo
    echo "Usage: $0 [COMMAND] [VERSION]"
    echo
    echo "Commands:"
    echo "  list                    List all available model versions"
    echo "  switch <version>        Switch to specific version (e.g., v2, v3)"
    echo "  current                 Show current active model"
    echo "  auto                    Auto-switch to next available version"
    echo "  help                    Show this help message"
    echo
    echo "Examples:"
    echo "  $0 list                 # List all models"
    echo "  $0 switch v2            # Switch to model v2"
    echo "  $0 current              # Show current active model"
    echo "  $0 auto                 # Auto-switch to next version"
    echo
    echo "Note: The active model is used by the inference system"
}

# Main script logic
case "${1:-list}" in
    "list")
        list_models
        ;;
    "switch")
        if [ -z "$2" ]; then
            print_error "Please specify a version to switch to"
            show_help
            exit 1
        fi
        switch_model "$2"
        ;;
    "current")
        show_current
        ;;
    "auto")
        auto_switch
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
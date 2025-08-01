#!/bin/bash

# CengBot Automatic Training Script
# This script automatically trains the model with the latest training data

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR="/home/ceng/cu_ceng_bot"
SRC_DIR="${BASE_DIR}/src"
LOGS_DIR="${BASE_DIR}/logs"
DATA_DIR="${BASE_DIR}/data"
MODELS_DIR="${BASE_DIR}/models"
VENV_DIR="${BASE_DIR}/venv"
TRAINING_SCRIPT="${SRC_DIR}/train_model.py"
DATASET_PATH="${DATA_DIR}/cengbot_qa_augmented.jsonl"
TRAINING_HISTORY_DIR="${LOGS_DIR}/training_history"

# Function to print colored output
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

# Function to check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "Command '$1' not found. Please install it first."
        exit 1
    fi
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "Checking GPU availability..."
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        
        # Check if RTX 5090 is available
        if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "5090"; then
            print_success "RTX 5090 detected - optimal for training"
        else
            print_warning "RTX 5090 not detected - training may be slower"
        fi
    else
        print_error "nvidia-smi not found. GPU training not available."
        exit 1
    fi
}

# Function to check required files
check_files() {
    print_status "Checking required files..."
    
    # Check training script
    if [ ! -f "$TRAINING_SCRIPT" ]; then
        print_error "Training script not found: $TRAINING_SCRIPT"
        exit 1
    fi
    
    # Check dataset
    if [ ! -f "$DATASET_PATH" ]; then
        print_error "Dataset not found: $DATASET_PATH"
        print_error "Please ensure the training data is available at: $DATASET_PATH"
        exit 1
    fi
    
    # Check .env file
    if [ ! -f "${BASE_DIR}/.env" ]; then
        print_error ".env file not found: ${BASE_DIR}/.env"
        print_error "Please create .env file with HUGGING_FACE_TOKEN"
        exit 1
    fi
    
    print_success "All required files found"
}

# Function to setup virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "${BASE_DIR}/requirements.txt" ]; then
        print_status "Installing requirements..."
        pip install -r "${BASE_DIR}/requirements.txt"
    else
        print_status "Installing essential packages..."
        pip install torch torchvision torchaudio transformers peft datasets accelerate bitsandbytes python-dotenv langdetect
    fi
    
    print_success "Virtual environment ready"
}

# Function to check dataset quality
check_dataset() {
    print_status "Checking dataset quality..."
    
    # Count lines in dataset
    line_count=$(wc -l < "$DATASET_PATH")
    print_status "Dataset contains $line_count training examples"
    
    if [ "$line_count" -lt 1000 ]; then
        print_warning "Dataset has fewer than 1000 examples - consider adding more data"
    fi
    
    # Check for valid JSON format
    if ! python3 -c "
import json
with open('$DATASET_PATH', 'r') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line.strip())
        except json.JSONDecodeError:
            print(f'Invalid JSON at line {i+1}')
            exit(1)
print('Dataset format is valid')
" 2>/dev/null; then
        print_error "Dataset contains invalid JSON format"
        exit 1
    fi
    
    print_success "Dataset quality check passed"
}

# Function to create backup
create_backup() {
    print_status "Creating backup of previous model..."
    
    if [ -d "$MODELS_DIR" ]; then
        backup_name="models_backup_$(date +%Y%m%d_%H%M%S)"
        cp -r "$MODELS_DIR" "${BASE_DIR}/${backup_name}"
        print_success "Backup created: ${backup_name}"
    else
        print_status "No previous model found - skipping backup"
    fi
}

# Function to get next version number
get_next_version() {
    mkdir -p "$TRAINING_HISTORY_DIR"
    
    # Find highest version number
    local highest_version=0
    for file in "$TRAINING_HISTORY_DIR"/v*.log; do
        if [ -f "$file" ]; then
            local version=$(basename "$file" .log | sed 's/v//')
            if [[ "$version" =~ ^[0-9]+$ ]] && [ "$version" -gt "$highest_version" ]; then
                highest_version="$version"
            fi
        fi
    done
    
    # Return next version
    echo $((highest_version + 1))
}

# Function to setup logging
setup_logging() {
    print_status "Setting up logging..."
    
    # Create logs directory
    mkdir -p "$LOGS_DIR/training"
    mkdir -p "$TRAINING_HISTORY_DIR"
    
    # Get next version number
    NEXT_VERSION=$(get_next_version)
    
    # Create log file with timestamp
    log_file="${LOGS_DIR}/training/auto_training_$(date +%Y%m%d_%H%M%S).log"
    version_log_file="${TRAINING_HISTORY_DIR}/v${NEXT_VERSION}.log"
    
    print_status "Training logs will be saved to:"
    print_status "  - Versioned log: $version_log_file"
    print_status "  - Timestamped log: $log_file"
    
    export TRAINING_LOG_FILE="$log_file"
    export TRAINING_VERSION="$NEXT_VERSION"
    export TRAINING_VERSION_LOG="$version_log_file"
}

# Function to run training
run_training() {
    print_header "Starting Model Training"
    
    # Change to base directory
    cd "$BASE_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Run training script
    print_status "Starting training process..."
    print_status "This may take several hours depending on your hardware..."
    
    if python3 "$TRAINING_SCRIPT" 2>&1 | tee -a "$TRAINING_LOG_FILE"; then
        print_success "Training completed successfully!"
    else
        print_error "Training failed! Check logs for details."
        exit 1
    fi
}

# Function to validate trained model
validate_model() {
    print_status "Validating trained model..."
    
    model_path="${MODELS_DIR}/cengbot-llama-3b-lora-v1/final-best-model-v1"
    
    if [ -d "$model_path" ]; then
        print_success "Model directory found: $model_path"
        
        # Check for essential files
        if [ -f "$model_path/config.json" ] && [ -f "$model_path/pytorch_model.bin" ]; then
            print_success "Model files validated successfully"
        else
            print_warning "Some model files may be missing"
        fi
    else
        print_error "Trained model not found at expected location"
        exit 1
    fi
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up temporary files..."
    
    # Remove any temporary files created during training
    find "$BASE_DIR" -name "*.tmp" -delete 2>/dev/null || true
    find "$BASE_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to show training summary
show_summary() {
    print_header "Training Summary"
    
    print_status "Training completed at: $(date)"
    print_status "Training version: v${TRAINING_VERSION}"
    print_status "Model saved to: ${MODELS_DIR}/cengbot-llama-3b-lora-v1/final-best-model-v1"
    print_status "Logs saved to:"
    print_status "  - Versioned log: $TRAINING_VERSION_LOG"
    print_status "  - Timestamped log: $TRAINING_LOG_FILE"
    
    # Show final loss if available
    if [ -f "${MODELS_DIR}/cengbot-llama-3b-lora-v1/final-best-model-v1/training_info.json" ]; then
        final_info=$(python3 -c "
import json
with open('${MODELS_DIR}/cengbot-llama-3b-lora-v1/final-best-model-v1/training_info.json', 'r') as f:
    data = json.load(f)
    print(f\"Version: {data.get('version', 'N/A')}\")
    print(f\"Final Loss: {data.get('final_loss', 'N/A')}\")
    print(f\"Total Steps: {data.get('total_steps', 'N/A')}\")
    print(f\"Dataset Size: {data.get('dataset_size', 'N/A')}\")
    print(f\"Learning Rate: {data.get('learning_rate', 'N/A')}\")
" 2>/dev/null || echo "Training info not available")
        print_status "$final_info"
    fi
    
    # Show training history
    print_status "Training history:"
    for version_file in "$TRAINING_HISTORY_DIR"/v*.log; do
        if [ -f "$version_file" ]; then
            version=$(basename "$version_file" .log)
            print_status "  - $version: $(wc -l < "$version_file") log lines"
        fi
    done
    
    print_success "Training process completed successfully!"
}

# Main function
main() {
    print_header "CengBot Automatic Training Script"
    print_status "Starting automatic training process..."
    
    # Check system requirements
    check_command "python3"
    check_command "nvidia-smi"
    
    # Run all checks and setup
    check_gpu
    check_files
    check_dataset
    setup_venv
    setup_logging
    create_backup
    
    # Run training
    run_training
    
    # Validate results
    validate_model
    
    # Cleanup and summary
    cleanup
    show_summary
    
    print_success "All tasks completed successfully!"
}

# Handle script interruption
trap 'print_error "Script interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"
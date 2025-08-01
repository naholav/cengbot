#!/bin/bash

# CengBot Environment Test Script
# Tests the environment setup without running actual training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

print_header "CengBot Environment Test"

# Test 1: Check directory structure
print_status "Testing directory structure..."
BASE_DIR="/home/ceng/cu_ceng_bot"
if [ -d "$BASE_DIR" ]; then
    print_success "Base directory exists: $BASE_DIR"
else
    print_error "Base directory not found: $BASE_DIR"
    exit 1
fi

# Test 2: Check training script
print_status "Testing training script..."
if [ -f "$BASE_DIR/src/train_model.py" ]; then
    print_success "Training script found"
else
    print_error "Training script not found: $BASE_DIR/src/train_model.py"
    exit 1
fi

# Test 3: Check .env file
print_status "Testing .env file..."
if [ -f "$BASE_DIR/.env" ]; then
    print_success ".env file found"
    
    # Check for required variables
    if grep -q "HUGGING_FACE_TOKEN" "$BASE_DIR/.env"; then
        print_success "HUGGING_FACE_TOKEN found in .env"
    else
        print_error "HUGGING_FACE_TOKEN not found in .env"
        exit 1
    fi
else
    print_error ".env file not found"
    exit 1
fi

# Test 4: Check Python
print_status "Testing Python installation..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    print_success "Python found: $python_version"
else
    print_error "Python3 not found"
    exit 1
fi

# Test 5: Check GPU
print_status "Testing GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
    print_success "GPU found: $gpu_info"
else
    print_error "nvidia-smi not found - GPU may not be available"
    exit 1
fi

# Test 6: Check dataset (optional)
print_status "Testing dataset availability..."
if [ -f "$BASE_DIR/data/cengbot_qa_augmented.jsonl" ]; then
    line_count=$(wc -l < "$BASE_DIR/data/cengbot_qa_augmented.jsonl")
    print_success "Dataset found with $line_count lines"
else
    print_error "Dataset not found: $BASE_DIR/data/cengbot_qa_augmented.jsonl"
    print_error "You need to create the dataset before training"
fi

# Test 7: Test Python imports
print_status "Testing Python dependencies..."
if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    print_success "PyTorch available"
else
    print_error "PyTorch not available - run training script to install dependencies"
fi

if python3 -c "import transformers; print('Transformers version:', transformers.__version__)" 2>/dev/null; then
    print_success "Transformers available"
else
    print_error "Transformers not available - run training script to install dependencies"
fi

print_header "Environment Test Complete"
print_success "Environment setup looks good!"
print_status "You can now run: ./scripts/train_model.sh"
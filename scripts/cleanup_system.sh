#!/bin/bash

# CengBot System Cleanup Script
# This script removes unnecessary files and cleans up the system

echo "🧹 CengBot System Cleanup"
echo "========================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get project root
PROJECT_ROOT="/home/ceng/cu_ceng_bot"

# Function to print colored output
print_status() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running from correct directory
if [ ! -f "$PROJECT_ROOT/scripts/cleanup_system.sh" ]; then
    print_error "Must be run from project root directory"
    exit 1
fi

cd "$PROJECT_ROOT"

# Clean Python cache files
print_status "Cleaning Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
print_success "Python cache files removed"

# Clean temporary files
print_status "Cleaning temporary files..."
find . -name "*.tmp" -o -name "*.temp" -o -name "*.bak" -o -name "*.old" -o -name "*~" -delete
print_success "Temporary files removed"

# Clean Windows-specific files (if any)
print_status "Cleaning Windows-specific files..."
find . -name "*.bat" -o -name "*.cmd" -o -name "*.exe" -o -name "*.msi" -o -name "*.dll" -o -name "Thumbs.db" -o -name "desktop.ini" -delete
print_success "Windows-specific files removed"

# Clean log files (optional)
if [ "$1" == "--clear-logs" ]; then
    print_status "Clearing log files..."
    if [ -d "logs" ]; then
        rm -f logs/*.log
        print_success "Log files cleared"
    else
        print_warning "No logs directory found"
    fi
fi

# Clean node_modules (optional)
if [ "$1" == "--clean-node-modules" ]; then
    print_status "Cleaning node_modules..."
    if [ -d "admin_frontend/node_modules" ]; then
        rm -rf admin_frontend/node_modules
        print_success "node_modules removed"
        print_warning "Run 'npm install' in admin_frontend/ to reinstall dependencies"
    else
        print_warning "No node_modules directory found"
    fi
fi

# Clean build directory (optional)
if [ "$1" == "--clean-build" ]; then
    print_status "Cleaning build directory..."
    if [ -d "admin_frontend/build" ]; then
        rm -rf admin_frontend/build
        print_success "Build directory removed"
        print_warning "Run 'npm run build' to rebuild the frontend"
    else
        print_warning "No build directory found"
    fi
fi

# Clean model cache (optional)
if [ "$1" == "--clean-model-cache" ]; then
    print_status "Cleaning model cache..."
    if [ -d "model_cache" ]; then
        print_warning "This will remove 6GB+ of cached models"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf model_cache
            print_success "Model cache removed"
            print_warning "Models will be re-downloaded on next use"
        else
            print_warning "Model cache cleanup cancelled"
        fi
    else
        print_warning "No model cache directory found"
    fi
fi

# Show disk usage
print_status "Current disk usage:"
du -sh . 2>/dev/null || echo "Unable to calculate disk usage"

echo ""
print_success "System cleanup completed!"
echo ""
echo "Available cleanup options:"
echo "  --clear-logs       : Clear all log files"
echo "  --clean-node-modules : Remove node_modules (812MB)"
echo "  --clean-build      : Remove build directory"
echo "  --clean-model-cache : Remove model cache (6GB+)"
echo ""
echo "Example: ./scripts/cleanup_system.sh --clear-logs"
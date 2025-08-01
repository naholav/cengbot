#!/bin/bash

echo "🔧 CengBot System Setup"
echo "======================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set working directory
cd "$(dirname "$0")/.."

echo -e "${BLUE}🚀 Setting up CengBot system for Ubuntu 22.04...${NC}"
echo ""

# Update system
echo -e "${BLUE}📦 Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo -e "${BLUE}📦 Installing system dependencies...${NC}"
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    nodejs \
    npm \
    rabbitmq-server \
    curl \
    wget \
    git \
    htop \
    dos2unix \
    build-essential

# Start and enable RabbitMQ
echo -e "${BLUE}🔄 Configuring RabbitMQ...${NC}"
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server

# Enable RabbitMQ management plugin
echo -e "${BLUE}🔧 Enabling RabbitMQ management plugin...${NC}"
sudo rabbitmq-plugins enable rabbitmq_management

# Create Python virtual environment
echo -e "${BLUE}🐍 Creating Python virtual environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo -e "${BLUE}📦 Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install -r config/requirements.txt

# Install Node.js dependencies
echo -e "${BLUE}📦 Installing Node.js dependencies...${NC}"
cd admin_frontend
npm install
cd ..

# Initialize database
echo -e "${BLUE}💾 Initializing database...${NC}"
python3 -c "from src.database_models import init_db; init_db()"

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p logs
mkdir -p config/env

# Fix script permissions
echo -e "${BLUE}🔧 Fixing script permissions...${NC}"
find scripts/ -name "*.sh" -exec chmod +x {} \;

# Create environment file template
echo -e "${BLUE}📝 Creating environment template...${NC}"
cat > .env.example << 'EOF'
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Database Configuration
DATABASE_URL=sqlite:///university_bot.db

# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=guest
RABBITMQ_PASSWORD=guest
RABBITMQ_QUEUE_NAME=cengbot_queue

# Model Configuration
BASE_MODEL_NAME=meta-llama/Llama-3.2-3B
LORA_MODEL_PATH=models/final-best-model-v1/method1
MODEL_TEMPERATURE=0.7
MODEL_MAX_NEW_TOKENS=200
USE_CUDA=true
MODEL_PRECISION=bfloat16

# API Configuration
API_HOST=0.0.0.0
API_PORT=8001
ADMIN_API_KEY=your_admin_api_key_here

# Frontend Configuration
FRONTEND_PORT=3000
FRONTEND_HOST=localhost

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/cengbot.log

# Training Configuration
AUTO_RETRAIN_THRESHOLD=1000
CLAUDE_API_KEY=your_claude_api_key_here
EOF

# Check if .env exists, if not copy from template
if [[ ! -f ".env" ]]; then
    echo -e "${BLUE}📝 Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️ Please edit .env file with your actual configuration values${NC}"
fi

echo ""
echo -e "${GREEN}🎉 CengBot Setup Complete!${NC}"
echo "========================="
echo ""
echo -e "${BLUE}📋 What was installed:${NC}"
echo -e "${GREEN}✅ Python 3 + pip + venv${NC}"
echo -e "${GREEN}✅ Node.js + npm${NC}"
echo -e "${GREEN}✅ RabbitMQ Server${NC}"
echo -e "${GREEN}✅ All Python dependencies${NC}"
echo -e "${GREEN}✅ All Node.js dependencies${NC}"
echo -e "${GREEN}✅ Database initialized${NC}"
echo -e "${GREEN}✅ Environment configuration${NC}"
echo ""
echo -e "${BLUE}🔧 Next steps:${NC}"
echo -e "${YELLOW}1. Edit .env file with your configuration:${NC}"
echo -e "${GREEN}   nano .env${NC}"
echo ""
echo -e "${YELLOW}2. Add your Telegram bot token:${NC}"
echo -e "${GREEN}   TELEGRAM_BOT_TOKEN=your_token_here${NC}"
echo ""
echo -e "${YELLOW}3. Start the system:${NC}"
echo -e "${GREEN}   ./scripts/start_system.sh${NC}"
echo ""
echo -e "${BLUE}🌐 Service URLs:${NC}"
echo -e "${GREEN}   Admin Panel: http://localhost:3000${NC}"
echo -e "${GREEN}   API Docs: http://localhost:8001/docs${NC}"
echo -e "${GREEN}   RabbitMQ Management: http://localhost:15672${NC}"
echo -e "${GREEN}   (Default login: guest/guest)${NC}"
echo ""
echo -e "${BLUE}📊 Useful commands:${NC}"
echo -e "${YELLOW}   ./scripts/health_check.sh    # Check system status${NC}"
echo -e "${YELLOW}   ./scripts/start_system.sh    # Start all services${NC}"
echo -e "${YELLOW}   ./scripts/stop_system.sh     # Stop all services${NC}"
echo ""
echo -e "${GREEN}✨ System is ready for production! 🚀${NC}"
echo ""
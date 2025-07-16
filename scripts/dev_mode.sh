#!/bin/bash

echo "💻 CengBot Development Mode"
echo "=========================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set working directory
cd "$(dirname "$0")/.."

echo -e "${BLUE}🔧 Starting CengBot in development mode...${NC}"
echo ""

# Create virtual environment if not exists
if [[ ! -d ".venv" ]]; then
    echo -e "${BLUE}🔧 Creating Python virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install -r config/requirements.txt

# Initialize database
echo -e "${BLUE}💾 Initializing database...${NC}"
python3 -c "from src.database_models import init_db; init_db()"

# Install frontend dependencies
echo -e "${BLUE}📦 Installing frontend dependencies...${NC}"
cd admin_frontend
npm install
cd ..

# Start RabbitMQ if not running
if ! systemctl is-active --quiet rabbitmq-server; then
    echo -e "${BLUE}🔄 Starting RabbitMQ...${NC}"
    sudo systemctl start rabbitmq-server
fi

# Create logs directory
mkdir -p logs

echo ""
echo -e "${GREEN}🎯 Development Environment Ready!${NC}"
echo "================================="
echo ""
echo -e "${BLUE}📝 Available commands:${NC}"
echo ""
echo -e "${YELLOW}1. AI Worker (Terminal 1):${NC}"
echo -e "${GREEN}   python3 src/ai_model_worker.py${NC}"
echo ""
echo -e "${YELLOW}2. Admin API (Terminal 2):${NC}"
echo -e "${GREEN}   python3 src/admin_rest_api.py${NC}"
echo ""
echo -e "${YELLOW}3. Telegram Bot (Terminal 3):${NC}"
echo -e "${GREEN}   python3 src/telegram_bot_rabbitmq.py${NC}"
echo ""
echo -e "${YELLOW}4. Frontend (Terminal 4):${NC}"
echo -e "${GREEN}   cd admin_frontend && npm start${NC}"
echo ""
echo -e "${YELLOW}5. Standalone Bot (Alternative):${NC}"
echo -e "${GREEN}   python3 src/telegram_bot_standalone.py${NC}"
echo ""
echo -e "${BLUE}🌐 URLs:${NC}"
echo -e "${GREEN}   Admin Panel: http://localhost:3000${NC}"
echo -e "${GREEN}   API Docs: http://localhost:8001/docs${NC}"
echo ""
echo -e "${BLUE}📊 Monitoring:${NC}"
echo -e "${YELLOW}   tail -f logs/worker.log${NC}"
echo -e "${YELLOW}   tail -f logs/admin.log${NC}"
echo -e "${YELLOW}   tail -f logs/bot.log${NC}"
echo ""
echo -e "${BLUE}🔧 Utils:${NC}"
echo -e "${YELLOW}   ./scripts/health_check.sh    # Check system status${NC}"
echo -e "${YELLOW}   ./scripts/start_system.sh    # Full auto start${NC}"
echo -e "${YELLOW}   ./scripts/stop_system.sh     # Stop all services${NC}"
echo ""
echo -e "${GREEN}💡 Happy coding! 🚀${NC}"
echo ""
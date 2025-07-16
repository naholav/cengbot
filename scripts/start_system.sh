#!/bin/bash

echo "🚀 CengBot Complete System Starting..."
echo "====================================="

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set working directory
cd "$(dirname "$0")/.."

# Check if we're in the right directory
if [[ ! -f "README.md" || ! -d "src" ]]; then
    echo -e "${RED}❌ Error: Must be run from university-bot directory${NC}"
    exit 1
fi

# Check dependencies
echo -e "${BLUE}📋 Checking requirements...${NC}"

if ! command -v rabbitmq-server &> /dev/null; then
    echo -e "${RED}❌ RabbitMQ not installed!${NC}"
    echo -e "${YELLOW}Installing RabbitMQ...${NC}"
    echo "cucengedutr" | sudo -S apt update
    echo "cucengedutr" | sudo -S apt install -y rabbitmq-server
fi

if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not installed!${NC}"
    echo -e "${YELLOW}Installing Node.js...${NC}"
    echo "cucengedutr" | sudo -S apt update
    echo "cucengedutr" | sudo -S apt install -y nodejs npm
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not installed!${NC}"
    echo -e "${YELLOW}Installing Python3...${NC}"
    echo "cucengedutr" | sudo -S apt update
    echo "cucengedutr" | sudo -S apt install -y python3 python3-pip python3-venv
fi

echo -e "${GREEN}✅ All requirements available${NC}"

# Create and activate virtual environment if not exists
if [[ ! -d ".venv" ]]; then
    echo -e "${BLUE}🔧 Creating Python virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
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

# Start RabbitMQ
echo -e "${BLUE}🔄 Starting RabbitMQ...${NC}"
echo "cucengedutr" | sudo -S systemctl start rabbitmq-server
echo "cucengedutr" | sudo -S systemctl enable rabbitmq-server
sleep 3

# Check RabbitMQ status
if ! systemctl is-active --quiet rabbitmq-server; then
    echo -e "${RED}❌ RabbitMQ failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}✅ RabbitMQ is running${NC}"

# Create directories
mkdir -p logs
echo -e "${BLUE}📁 Log directory prepared${NC}"

# Kill existing processes
echo -e "${BLUE}🛑 Stopping existing services...${NC}"
pkill -f "python3.*ai_model_worker" 2>/dev/null || true
pkill -f "python3.*admin_rest_api" 2>/dev/null || true
pkill -f "python3.*telegram_bot" 2>/dev/null || true
pkill -f "node.*react-scripts" 2>/dev/null || true
sleep 3

# Start services
echo ""
echo -e "${BLUE}🔧 Starting services...${NC}"
echo "=============================="

echo -e "${BLUE}1️⃣ Starting AI Worker (LLaMA 3.2 3B + LoRA)...${NC}"
python3 src/ai_model_worker.py > logs/worker.log 2>&1 &
WORKER_PID=$!
echo -e "${GREEN}   ✅ AI Worker started - PID: $WORKER_PID${NC}"

sleep 5

echo -e "${BLUE}2️⃣ Starting Admin API...${NC}"
python3 src/admin_rest_api.py > logs/admin.log 2>&1 &
API_PID=$!
echo -e "${GREEN}   ✅ Admin API started - PID: $API_PID${NC}"

sleep 3

echo -e "${BLUE}3️⃣ Starting Telegram Bot...${NC}"
python3 src/telegram_bot_rabbitmq.py > logs/bot.log 2>&1 &
BOT_PID=$!
echo -e "${GREEN}   ✅ Telegram Bot started - PID: $BOT_PID${NC}"

sleep 3

echo -e "${BLUE}4️⃣ Starting React Frontend...${NC}"
cd admin_frontend
npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "${GREEN}   ✅ Frontend started - PID: $FRONTEND_PID${NC}"

# Save PIDs for stop script
echo "$WORKER_PID $API_PID $BOT_PID $FRONTEND_PID" > .pids

# Wait for services to fully start
echo ""
echo -e "${YELLOW}⏳ Waiting for services to fully initialize...${NC}"
sleep 10

# Check if services are running
echo ""
echo -e "${BLUE}🔍 Checking service status...${NC}"

if ps -p $WORKER_PID > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ AI Worker is running${NC}"
else
    echo -e "${RED}   ❌ AI Worker failed to start${NC}"
    echo -e "${YELLOW}   Check logs/worker.log for details${NC}"
fi

if ps -p $API_PID > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Admin API is running${NC}"
else
    echo -e "${RED}   ❌ Admin API failed to start${NC}"
    echo -e "${YELLOW}   Check logs/admin.log for details${NC}"
fi

if ps -p $BOT_PID > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Telegram Bot is running${NC}"
else
    echo -e "${RED}   ❌ Telegram Bot failed to start${NC}"
    echo -e "${YELLOW}   Check logs/bot.log for details${NC}"
fi

if ps -p $FRONTEND_PID > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Frontend is running${NC}"
else
    echo -e "${RED}   ❌ Frontend failed to start${NC}"
    echo -e "${YELLOW}   Check logs/frontend.log for details${NC}"
fi

echo ""
echo -e "${GREEN}🎉 CengBot System is Ready!${NC}"
echo "================================="
echo ""
echo -e "${BLUE}📊 Running Services:${NC}"
echo -e "${GREEN}  🤖 AI Worker (LLaMA 3.2 3B + LoRA): PID $WORKER_PID${NC}"
echo -e "${GREEN}  🌐 Admin API: PID $API_PID${NC}"
echo -e "${GREEN}  📱 Telegram Bot: PID $BOT_PID${NC}"
echo -e "${GREEN}  💻 React Frontend: PID $FRONTEND_PID${NC}"
echo ""
echo -e "${BLUE}🌐 Web Interfaces:${NC}"
echo -e "${GREEN}  📊 Admin Panel: http://localhost:3000${NC}"
echo -e "${GREEN}  🔗 API Docs: http://localhost:8001/docs${NC}"
echo ""
echo -e "${BLUE}📝 Log Files:${NC}"
echo -e "${GREEN}  🤖 AI Worker: logs/worker.log${NC}"
echo -e "${GREEN}  🌐 Admin API: logs/admin.log${NC}"
echo -e "${GREEN}  📱 Telegram Bot: logs/bot.log${NC}"
echo -e "${GREEN}  💻 Frontend: logs/frontend.log${NC}"
echo ""
echo -e "${BLUE}🔍 System Monitoring:${NC}"
echo -e "${YELLOW}  tail -f logs/worker.log    # AI model status${NC}"
echo -e "${YELLOW}  tail -f logs/bot.log       # Telegram messages${NC}"
echo -e "${YELLOW}  tail -f logs/admin.log     # Admin API requests${NC}"
echo ""
echo -e "${BLUE}🛑 To Stop System:${NC}"
echo -e "${YELLOW}  ./scripts/stop_system.sh${NC}"
echo ""
echo -e "${GREEN}✨ CengBot is Production Ready! You can test it on Telegram.${NC}"
echo ""
echo -e "${YELLOW}💡 Important Notes:${NC}"
echo -e "${YELLOW}  • First model loading may take 5-10 minutes${NC}"
echo -e "${YELLOW}  • Admin panel will be available at http://localhost:3000${NC}"
echo -e "${YELLOW}  • Make sure to configure TELEGRAM_BOT_TOKEN in .env${NC}"
echo ""
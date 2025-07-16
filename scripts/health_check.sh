#!/bin/bash

echo "🔍 CengBot System Health Check"
echo "=============================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set working directory
cd "$(dirname "$0")/.."

echo -e "${BLUE}🔍 Checking system dependencies...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python3 not found${NC}"
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>&1)
    echo -e "${GREEN}✅ Node.js: $NODE_VERSION${NC}"
else
    echo -e "${RED}❌ Node.js not found${NC}"
fi

# Check RabbitMQ
if command -v rabbitmq-server &> /dev/null; then
    if systemctl is-active --quiet rabbitmq-server; then
        echo -e "${GREEN}✅ RabbitMQ: Running${NC}"
    else
        echo -e "${YELLOW}⚠️ RabbitMQ: Installed but not running${NC}"
    fi
else
    echo -e "${RED}❌ RabbitMQ not found${NC}"
fi

echo ""
echo -e "${BLUE}🔍 Checking CengBot services...${NC}"

# Check AI Worker
if pgrep -f "python3.*ai_model_worker" > /dev/null; then
    AI_PID=$(pgrep -f "python3.*ai_model_worker")
    echo -e "${GREEN}✅ AI Worker: Running (PID: $AI_PID)${NC}"
else
    echo -e "${RED}❌ AI Worker: Not running${NC}"
fi

# Check Admin API
if pgrep -f "python3.*admin_rest_api" > /dev/null; then
    API_PID=$(pgrep -f "python3.*admin_rest_api")
    echo -e "${GREEN}✅ Admin API: Running (PID: $API_PID)${NC}"
    
    # Check if API is responding
    if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}  📡 API Health: OK${NC}"
    else
        echo -e "${YELLOW}  📡 API Health: Not responding${NC}"
    fi
else
    echo -e "${RED}❌ Admin API: Not running${NC}"
fi

# Check Telegram Bot
if pgrep -f "python3.*telegram_bot" > /dev/null; then
    BOT_PID=$(pgrep -f "python3.*telegram_bot")
    echo -e "${GREEN}✅ Telegram Bot: Running (PID: $BOT_PID)${NC}"
else
    echo -e "${RED}❌ Telegram Bot: Not running${NC}"
fi

# Check Frontend
if pgrep -f "node.*react-scripts" > /dev/null; then
    FRONTEND_PID=$(pgrep -f "node.*react-scripts")
    echo -e "${GREEN}✅ Frontend: Running (PID: $FRONTEND_PID)${NC}"
    
    # Check if frontend is responding
    if curl -s -f http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}  📡 Frontend Health: OK${NC}"
    else
        echo -e "${YELLOW}  📡 Frontend Health: Not responding${NC}"
    fi
else
    echo -e "${RED}❌ Frontend: Not running${NC}"
fi

echo ""
echo -e "${BLUE}🔍 Checking system resources...${NC}"

# Check disk space
DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
if [[ $DISK_USAGE -lt 80 ]]; then
    echo -e "${GREEN}✅ Disk Usage: ${DISK_USAGE}%${NC}"
else
    echo -e "${YELLOW}⚠️ Disk Usage: ${DISK_USAGE}% (High)${NC}"
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo -e "${GREEN}✅ Memory Usage: ${MEM_USAGE}%${NC}"

# Check database
if [[ -f "university_bot.db" ]]; then
    DB_SIZE=$(du -h university_bot.db | cut -f1)
    echo -e "${GREEN}✅ Database: ${DB_SIZE}${NC}"
else
    echo -e "${RED}❌ Database: Not found${NC}"
fi

echo ""
echo -e "${BLUE}🔍 Checking log files...${NC}"

for logfile in logs/*.log; do
    if [[ -f "$logfile" ]]; then
        LOG_SIZE=$(du -h "$logfile" | cut -f1)
        LOG_LINES=$(wc -l < "$logfile")
        echo -e "${GREEN}✅ $(basename "$logfile"): ${LOG_SIZE} (${LOG_LINES} lines)${NC}"
    fi
done

echo ""
echo -e "${BLUE}🔍 Network connectivity...${NC}"

# Check ports
if ss -tuln | grep -q ":8001 "; then
    echo -e "${GREEN}✅ Port 8001: Open (Admin API)${NC}"
else
    echo -e "${RED}❌ Port 8001: Closed${NC}"
fi

if ss -tuln | grep -q ":3000 "; then
    echo -e "${GREEN}✅ Port 3000: Open (Frontend)${NC}"
else
    echo -e "${RED}❌ Port 3000: Closed${NC}"
fi

if ss -tuln | grep -q ":5672 "; then
    echo -e "${GREEN}✅ Port 5672: Open (RabbitMQ)${NC}"
else
    echo -e "${RED}❌ Port 5672: Closed${NC}"
fi

echo ""
echo -e "${BLUE}📊 System Summary${NC}"
echo "================="

# Count running services
RUNNING_SERVICES=0
if pgrep -f "python3.*ai_model_worker" > /dev/null; then ((RUNNING_SERVICES++)); fi
if pgrep -f "python3.*admin_rest_api" > /dev/null; then ((RUNNING_SERVICES++)); fi
if pgrep -f "python3.*telegram_bot" > /dev/null; then ((RUNNING_SERVICES++)); fi
if pgrep -f "node.*react-scripts" > /dev/null; then ((RUNNING_SERVICES++)); fi

if [[ $RUNNING_SERVICES -eq 4 ]]; then
    echo -e "${GREEN}🎉 All services are running! (4/4)${NC}"
    echo -e "${GREEN}📊 Admin Panel: http://localhost:3000${NC}"
    echo -e "${GREEN}🔗 API Docs: http://localhost:8001/docs${NC}"
elif [[ $RUNNING_SERVICES -gt 0 ]]; then
    echo -e "${YELLOW}⚠️ Partial system running ($RUNNING_SERVICES/4 services)${NC}"
    echo -e "${YELLOW}💡 Try: ./scripts/start_system.sh${NC}"
else
    echo -e "${RED}❌ System is not running (0/4 services)${NC}"
    echo -e "${YELLOW}💡 Start with: ./scripts/start_system.sh${NC}"
fi

echo ""
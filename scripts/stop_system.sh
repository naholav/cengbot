#!/bin/bash

echo "🛑 CengBot System Stopping..."
echo "============================="

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set working directory
cd "$(dirname "$0")/.."

# Check for PID file
if [[ -f ".pids" ]]; then
    echo -e "${BLUE}📋 Reading saved PIDs...${NC}"
    read -r WORKER_PID API_PID BOT_PID FRONTEND_PID < .pids
    
    # Stop services by PID
    echo -e "${BLUE}🛑 Stopping services by PID...${NC}"
    
    if [[ -n "$WORKER_PID" ]] && ps -p $WORKER_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}  Stopping AI Worker (PID: $WORKER_PID)...${NC}"
        kill $WORKER_PID
    fi
    
    if [[ -n "$API_PID" ]] && ps -p $API_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}  Stopping Admin API (PID: $API_PID)...${NC}"
        kill $API_PID
    fi
    
    if [[ -n "$BOT_PID" ]] && ps -p $BOT_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}  Stopping Telegram Bot (PID: $BOT_PID)...${NC}"
        kill $BOT_PID
    fi
    
    if [[ -n "$FRONTEND_PID" ]] && ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}  Stopping Frontend (PID: $FRONTEND_PID)...${NC}"
        kill $FRONTEND_PID
    fi
    
    sleep 3
    
    # Force kill if still running
    if [[ -n "$WORKER_PID" ]] && ps -p $WORKER_PID > /dev/null 2>&1; then
        echo -e "${RED}  Force killing AI Worker...${NC}"
        kill -9 $WORKER_PID
    fi
    
    if [[ -n "$API_PID" ]] && ps -p $API_PID > /dev/null 2>&1; then
        echo -e "${RED}  Force killing Admin API...${NC}"
        kill -9 $API_PID
    fi
    
    if [[ -n "$BOT_PID" ]] && ps -p $BOT_PID > /dev/null 2>&1; then
        echo -e "${RED}  Force killing Telegram Bot...${NC}"
        kill -9 $BOT_PID
    fi
    
    if [[ -n "$FRONTEND_PID" ]] && ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "${RED}  Force killing Frontend...${NC}"
        kill -9 $FRONTEND_PID
    fi
    
    rm .pids
    echo -e "${GREEN}✅ PID file cleaned${NC}"
else
    echo -e "${YELLOW}⚠️ No PID file found, using process names...${NC}"
fi

# Kill by process name patterns
echo -e "${BLUE}🔍 Killing remaining processes...${NC}"
pkill -f "python3.*ai_model_worker" 2>/dev/null || true
pkill -f "python3.*admin_rest_api" 2>/dev/null || true
pkill -f "python3.*telegram_bot" 2>/dev/null || true
pkill -f "node.*react-scripts" 2>/dev/null || true

# Kill any remaining CengBot processes
pkill -f "python3.*src/" 2>/dev/null || true

sleep 2

# Check if processes are still running
REMAINING=$(ps aux | grep -E "(python3.*(ai_model_worker|admin_rest_api|telegram_bot)|node.*react-scripts)" | grep -v grep | wc -l)

if [[ $REMAINING -eq 0 ]]; then
    echo -e "${GREEN}✅ All CengBot services stopped successfully${NC}"
else
    echo -e "${YELLOW}⚠️ Some processes may still be running${NC}"
    echo -e "${YELLOW}   Check with: ps aux | grep -E 'python3.*src|node.*react-scripts'${NC}"
fi

echo ""
echo -e "${GREEN}🎉 CengBot System Stopped!${NC}"
echo "========================="
echo ""
echo -e "${BLUE}📊 To restart the system:${NC}"
echo -e "${YELLOW}  ./scripts/start_system.sh${NC}"
echo ""
echo -e "${BLUE}📝 Log files are preserved in logs/ directory${NC}"
echo ""
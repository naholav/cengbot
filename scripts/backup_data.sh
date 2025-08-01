#!/bin/bash
#
# CengBot Data Backup Script
# ==========================
# This script creates a backup of all important data including:
# - SQLite database
# - Training data files
# - Model files
# - Logs
#
# Author: naholav
# Date: August 2025

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define backup directory with timestamp
BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$PROJECT_ROOT/backups/backup_$BACKUP_TIMESTAMP"

echo -e "${GREEN}🔄 CengBot Data Backup Script${NC}"
echo -e "${GREEN}==============================${NC}"
echo "Creating backup at: $BACKUP_DIR"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
echo -e "\n${YELLOW}📦 Backing up database...${NC}"
if [ -f "$PROJECT_ROOT/university_bot.db" ]; then
    cp "$PROJECT_ROOT/university_bot.db" "$BACKUP_DIR/"
    echo -e "${GREEN}✓ Database backed up${NC}"
else
    echo -e "${RED}✗ Database not found${NC}"
fi

# Backup training data
echo -e "\n${YELLOW}📦 Backing up training data...${NC}"
if [ -d "$PROJECT_ROOT/data" ]; then
    cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/"
    echo -e "${GREEN}✓ Training data backed up${NC}"
else
    echo -e "${RED}✗ Training data directory not found${NC}"
fi

# Backup models
echo -e "\n${YELLOW}📦 Backing up models...${NC}"
if [ -d "$PROJECT_ROOT/models" ]; then
    cp -r "$PROJECT_ROOT/models" "$BACKUP_DIR/"
    echo -e "${GREEN}✓ Models backed up${NC}"
else
    echo -e "${RED}✗ Models directory not found${NC}"
fi

# Backup logs
echo -e "\n${YELLOW}📦 Backing up logs...${NC}"
if [ -d "$PROJECT_ROOT/logs" ]; then
    cp -r "$PROJECT_ROOT/logs" "$BACKUP_DIR/"
    echo -e "${GREEN}✓ Logs backed up${NC}"
else
    echo -e "${RED}✗ Logs directory not found${NC}"
fi

# Backup environment file (without sensitive data)
echo -e "\n${YELLOW}📦 Backing up environment template...${NC}"
if [ -f "$PROJECT_ROOT/.env.example" ]; then
    cp "$PROJECT_ROOT/.env.example" "$BACKUP_DIR/"
    echo -e "${GREEN}✓ Environment template backed up${NC}"
fi

# Create backup info file
echo -e "\n${YELLOW}📝 Creating backup info...${NC}"
cat > "$BACKUP_DIR/backup_info.txt" <<EOF
CengBot Backup Information
==========================
Backup Date: $(date)
Backup Directory: $BACKUP_DIR
Project Root: $PROJECT_ROOT

Contents:
- Database: university_bot.db
- Training Data: data/
- Models: models/
- Logs: logs/
- Environment Template: .env.example

To restore:
1. Copy files back to their original locations
2. Ensure proper permissions are set
3. Restart all services
EOF

echo -e "${GREEN}✓ Backup info created${NC}"

# Create compressed archive
echo -e "\n${YELLOW}🗜️ Creating compressed archive...${NC}"
cd "$PROJECT_ROOT/backups"
tar -czf "backup_$BACKUP_TIMESTAMP.tar.gz" "backup_$BACKUP_TIMESTAMP"
echo -e "${GREEN}✓ Archive created: backup_$BACKUP_TIMESTAMP.tar.gz${NC}"

# Calculate sizes
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
ARCHIVE_SIZE=$(du -sh "backup_$BACKUP_TIMESTAMP.tar.gz" | cut -f1)

echo -e "\n${GREEN}✅ Backup completed successfully!${NC}"
echo -e "Backup directory: $BACKUP_DIR ($BACKUP_SIZE)"
echo -e "Archive file: $PROJECT_ROOT/backups/backup_$BACKUP_TIMESTAMP.tar.gz ($ARCHIVE_SIZE)"

# Cleanup old backups (keep last 5)
echo -e "\n${YELLOW}🧹 Cleaning up old backups...${NC}"
cd "$PROJECT_ROOT/backups"
ls -t backup_*.tar.gz 2>/dev/null | tail -n +6 | xargs -r rm -f
REMAINING_BACKUPS=$(ls backup_*.tar.gz 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Kept $REMAINING_BACKUPS most recent backups${NC}"
# 🚀 CengBot Deployment Guide

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB available space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Network**: Stable internet connection

### Software Dependencies
- **Python**: 3.8 or higher
- **Node.js**: 16.x or higher
- **npm**: 8.x or higher
- **RabbitMQ**: 3.11+
- **Git**: Latest version

## 🛠️ Installation Methods

### Method 1: Automated Setup (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/your-username/university-bot.git
cd university-bot

# 2. Run automated setup
chmod +x scripts/setup_system.sh
./scripts/setup_system.sh

# 3. Configure environment
cp config/env/.env.production .env
nano .env  # Edit with your values

# 4. Start system
./scripts/start_system.sh
```

### Method 2: Manual Setup

#### Step 1: System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y nodejs npm
sudo apt install -y rabbitmq-server
sudo apt install -y git curl wget

# CentOS/RHEL
sudo yum update
sudo yum install -y python3 python3-pip
sudo yum install -y nodejs npm
sudo yum install -y rabbitmq-server
sudo yum install -y git curl wget

# Enable and start RabbitMQ
sudo systemctl enable rabbitmq-server
sudo systemctl start rabbitmq-server
```

#### Step 2: Python Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r config/requirements.txt
```

#### Step 3: Frontend Setup
```bash
# Install Node.js dependencies
cd admin_frontend
npm install
npm run build  # For production
cd ..
```

#### Step 4: Database Setup
```bash
# Initialize SQLite database
cd src
python3 -c "from database_models import init_db; init_db()"
cd ..
```

#### Step 5: Configuration
```bash
# Create environment file
cp config/env/.env.production .env

# Edit configuration
nano .env
```

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TELEGRAM_TOPIC_ID=                    # Optional: specific topic ID

# AI Model Configuration
BASE_MODEL_NAME=meta-llama/Llama-3.2-3B
LORA_MODEL_PATH=models/final-best-model-v1/method1
MODEL_TEMPERATURE=0.7
MODEL_MAX_NEW_TOKENS=200
USE_CUDA=true                         # Set to false if no GPU
MODEL_PRECISION=bfloat16

# Database Configuration
DATABASE_URL=sqlite:///university_bot.db

# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_URL=amqp://localhost:5672
QUESTIONS_QUEUE=questions
ANSWERS_QUEUE=answers

# API Server Configuration
API_HOST=0.0.0.0
API_PORT=8001
CORS_ORIGINS=http://localhost:3000,http://your-domain.com

# System Configuration
MAX_CONCURRENT_REQUESTS=3
LOG_LEVEL=INFO
```

### Bot Token Setup
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Create new bot: `/newbot`
3. Follow instructions to get your bot token
4. Add token to `.env` file

## 🐳 Docker Deployment (Optional)

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY config/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8001

# Run application
CMD ["python", "src/telegram_bot.py"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  rabbitmq:
    image: rabbitmq:3.11-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: password

  cengbot:
    build: .
    depends_on:
      - rabbitmq
    environment:
      RABBITMQ_URL: amqp://admin:password@rabbitmq:5672
    volumes:
      - ./logs:/app/logs
      - ./university_bot.db:/app/university_bot.db
    restart: unless-stopped

  admin-api:
    build: .
    command: ["uvicorn", "admin_rest_api:app", "--host", "0.0.0.0", "--port", "8001"]
    ports:
      - "8001:8001"
    depends_on:
      - rabbitmq
    volumes:
      - ./logs:/app/logs
      - ./university_bot.db:/app/university_bot.db
    restart: unless-stopped

  frontend:
    build:
      context: ./admin_frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - admin-api
    restart: unless-stopped
```

## 🔧 Production Deployment

### Step 1: Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y nodejs npm
sudo apt install -y rabbitmq-server nginx
sudo apt install -y supervisor  # For process management

# Configure firewall
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### Step 2: Application Deployment
```bash
# Clone to production directory
sudo mkdir -p /opt/cengbot
sudo chown $USER:$USER /opt/cengbot
cd /opt/cengbot

git clone https://github.com/your-username/university-bot.git .
chmod +x scripts/*.sh
./scripts/setup_system.sh
```

### Step 3: Process Management with Supervisor
```ini
# /etc/supervisor/conf.d/cengbot.conf
[group:cengbot]
programs=cengbot-worker,cengbot-bot,cengbot-api

[program:cengbot-worker]
directory=/opt/cengbot
command=/opt/cengbot/.venv/bin/python src/ai_model_worker.py
user=cengbot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/cengbot/logs/worker.log

[program:cengbot-bot]
directory=/opt/cengbot
command=/opt/cengbot/.venv/bin/python src/telegram_bot_rabbitmq.py
user=cengbot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/cengbot/logs/bot.log

[program:cengbot-api]
directory=/opt/cengbot
command=/opt/cengbot/.venv/bin/uvicorn admin_rest_api:app --host 0.0.0.0 --port 8001
user=cengbot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/cengbot/logs/admin.log
```

### Step 4: Nginx Configuration
```nginx
# /etc/nginx/sites-available/cengbot
server {
    listen 80;
    server_name your-domain.com;

    # Admin API
    location /api/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Frontend
    location / {
        root /opt/cengbot/admin_frontend/build;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
}
```

### Step 5: SSL/TLS Setup with Let's Encrypt
```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring & Maintenance

### System Monitoring
```bash
# Check system health
./scripts/health_check.sh

# Monitor logs
tail -f logs/worker.log
tail -f logs/bot.log
tail -f logs/admin.log

# Check process status
sudo supervisorctl status cengbot:*
```

### Database Backup
```bash
# Create backup script
#!/bin/bash
BACKUP_DIR="/opt/cengbot/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
sqlite3 /opt/cengbot/university_bot.db ".backup $BACKUP_DIR/university_bot_$DATE.db"

# Keep only last 7 days
find $BACKUP_DIR -name "university_bot_*.db" -mtime +7 -delete
```

### Log Rotation
```bash
# /etc/logrotate.d/cengbot
/opt/cengbot/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 cengbot cengbot
    postrotate
        sudo supervisorctl restart cengbot:*
    endscript
}
```

## 🔒 Security Hardening

### User Security
```bash
# Create dedicated user
sudo useradd -r -s /bin/false cengbot
sudo chown -R cengbot:cengbot /opt/cengbot
sudo chmod -R 755 /opt/cengbot
sudo chmod 600 /opt/cengbot/.env
```

### File Permissions
```bash
# Secure configuration files
chmod 600 /opt/cengbot/.env
chmod 600 /opt/cengbot/config/env/.env.production

# Secure database
chmod 640 /opt/cengbot/university_bot.db
chown cengbot:cengbot /opt/cengbot/university_bot.db
```

### Network Security
```bash
# Configure iptables
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -A INPUT -j DROP

# Save rules
sudo iptables-save > /etc/iptables/rules.v4
```

## 🐛 Troubleshooting

### Common Issues

#### Model Loading Problems
```bash
# Check CUDA availability
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Check model files
ls -la models/final-best-model-v1/method1/

# Monitor memory usage
free -h
```

#### Service Failures
```bash
# Check supervisor status
sudo supervisorctl status

# Restart services
sudo supervisorctl restart cengbot:*

# Check logs
sudo supervisorctl tail -f cengbot-worker stderr
```

#### Database Issues
```bash
# Check database integrity
sqlite3 university_bot.db "PRAGMA integrity_check;"

# Fix permissions
sudo chown cengbot:cengbot university_bot.db
sudo chmod 640 university_bot.db
```

## 📈 Performance Optimization

### Model Optimization
```bash
# Enable quantization
USE_4BIT_QUANTIZATION=true
MODEL_PRECISION=bfloat16

# Adjust concurrent requests
MAX_CONCURRENT_REQUESTS=3
```

### Database Optimization
```sql
-- Add indexes for better performance
CREATE INDEX idx_raw_data_telegram_id ON raw_data(telegram_id);
CREATE INDEX idx_raw_data_created_at ON raw_data(created_at);
CREATE INDEX idx_raw_data_language ON raw_data(language);
```

## 🔄 Updates & Maintenance

### Update Process
```bash
# 1. Backup data
./scripts/backup_data.sh

# 2. Stop services
sudo supervisorctl stop cengbot:*

# 3. Update code
git pull origin main

# 4. Update dependencies
source .venv/bin/activate
pip install -r config/requirements.txt

# 5. Run migrations (if any)
cd src
python3 -c "from database_models import init_db; init_db()"

# 6. Start services
sudo supervisorctl start cengbot:*
```

### Health Checks
```bash
# Automated health check
#!/bin/bash
if ! ./scripts/health_check.sh; then
    echo "System unhealthy, restarting services..."
    sudo supervisorctl restart cengbot:*
    
    # Send alert
    curl -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
         -d "chat_id=$ADMIN_CHAT_ID" \
         -d "text=CengBot system restarted due to health check failure"
fi
```

This deployment guide provides a comprehensive approach to deploying CengBot in production environments with proper security, monitoring, and maintenance procedures.
#!/bin/bash

# Production Deployment Script for CengBot
# Deploys CengBot to production environment with all optimizations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_USER="ceng"
BACKUP_DIR="/home/ceng/cu_ceng_bot_backup"

# Logging
LOG_FILE="$PROJECT_DIR/logs/deployment.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

info() {
    log "INFO: $1"
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    log "SUCCESS: $1"
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    log "WARNING: $1"
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    log "ERROR: $1"
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for production deployment"
    fi
}

# Create backup of existing deployment
create_backup() {
    info "Creating backup of existing deployment..."
    
    if [[ -d "$BACKUP_DIR" ]]; then
        rm -rf "$BACKUP_DIR"
    fi
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Backup important files
    cp -r "$PROJECT_DIR" "$BACKUP_DIR/cu_ceng_bot_$(date +%Y%m%d_%H%M%S)"
    
    # Backup systemd services
    if [[ -f "/etc/systemd/system/cengbot-worker.service" ]]; then
        cp /etc/systemd/system/cengbot-*.service "$BACKUP_DIR/" 2>/dev/null || true
    fi
    
    success "Backup created at $BACKUP_DIR"
}

# Stop existing services
stop_services() {
    info "Stopping existing services..."
    
    local services=("cengbot-worker" "cengbot-bot" "cengbot-admin" "cengbot-monitor")
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            systemctl stop "$service"
            info "Stopped $service"
        fi
    done
    
    success "All services stopped"
}

# Install system dependencies
install_dependencies() {
    info "Installing system dependencies..."
    
    # Update system
    apt-get update
    
    # Install required packages
    apt-get install -y \
        build-essential \
        curl \
        git \
        nginx \
        postgresql-client \
        redis-server \
        rabbitmq-server \
        htop \
        iotop \
        nethogs \
        sysstat \
        nmon \
        tmux \
        vim \
        wget \
        zip \
        unzip \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release
    
    # Start and enable services
    systemctl start rabbitmq-server
    systemctl enable rabbitmq-server
    
    systemctl start redis-server
    systemctl enable redis-server
    
    success "System dependencies installed"
}

# Setup nginx reverse proxy
setup_nginx() {
    info "Setting up nginx reverse proxy..."
    
    # Create nginx configuration
    cat > /etc/nginx/sites-available/cengbot << EOF
server {
    listen 80;
    server_name localhost;
    
    # Admin API
    location /api/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization' always;
        
        if (\$request_method = OPTIONS) {
            return 204;
        }
    }
    
    # Frontend
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_cache_bypass \$http_upgrade;
    }
    
    # Static files
    location /static/ {
        alias $PROJECT_DIR/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check
    location /health {
        proxy_pass http://127.0.0.1:8001/health;
        access_log off;
    }
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
}
EOF
    
    # Enable site
    ln -sf /etc/nginx/sites-available/cengbot /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test nginx configuration
    nginx -t
    
    # Restart nginx
    systemctl restart nginx
    systemctl enable nginx
    
    success "Nginx reverse proxy configured"
}

# Setup log rotation
setup_log_rotation() {
    info "Setting up log rotation..."
    
    cat > /etc/logrotate.d/cengbot << EOF
$PROJECT_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ceng ceng
    postrotate
        systemctl reload cengbot-worker cengbot-bot cengbot-admin cengbot-monitor 2>/dev/null || true
    endscript
}

$PROJECT_DIR/logs/monitoring/*.jsonl {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ceng ceng
}

$PROJECT_DIR/logs/training/*.log {
    weekly
    rotate 12
    compress
    delaycompress
    missingok
    notifempty
    create 0644 ceng ceng
}
EOF
    
    success "Log rotation configured"
}

# Setup monitoring
setup_monitoring() {
    info "Setting up system monitoring..."
    
    # Create monitoring script
    cat > /usr/local/bin/cengbot-health.sh << EOF
#!/bin/bash
# CengBot health monitoring script

# Check services
services=("cengbot-worker" "cengbot-bot" "cengbot-admin" "rabbitmq-server" "nginx")
for service in "\${services[@]}"; do
    if ! systemctl is-active --quiet "\$service"; then
        echo "ERROR: \$service is not running"
        systemctl start "\$service"
    fi
done

# Check disk space
disk_usage=\$(df / | tail -1 | awk '{print \$5}' | sed 's/%//')
if [[ \$disk_usage -gt 90 ]]; then
    echo "WARNING: Disk usage is \${disk_usage}%"
fi

# Check memory usage
memory_usage=\$(free | grep Mem | awk '{printf "%.0f", \$3/\$2 * 100}')
if [[ \$memory_usage -gt 85 ]]; then
    echo "WARNING: Memory usage is \${memory_usage}%"
fi

# Check API health
if ! curl -s http://localhost:8001/health > /dev/null; then
    echo "ERROR: Admin API is not responding"
    systemctl restart cengbot-admin
fi
EOF
    
    chmod +x /usr/local/bin/cengbot-health.sh
    
    # Create cron job for health monitoring
    cat > /etc/cron.d/cengbot-health << EOF
# CengBot health monitoring
*/5 * * * * root /usr/local/bin/cengbot-health.sh >> /var/log/cengbot-health.log 2>&1
EOF
    
    success "System monitoring configured"
}

# Setup firewall
setup_firewall() {
    info "Setting up firewall..."
    
    # Install ufw if not present
    apt-get install -y ufw
    
    # Reset firewall
    ufw --force reset
    
    # Default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH (be careful!)
    ufw allow ssh
    
    # Allow HTTP/HTTPS
    ufw allow 80/tcp
    ufw allow 443/tcp
    
    # Allow internal services
    ufw allow from 127.0.0.1 to any port 3000  # Frontend
    ufw allow from 127.0.0.1 to any port 8001  # Admin API
    ufw allow from 127.0.0.1 to any port 5672  # RabbitMQ
    ufw allow from 127.0.0.1 to any port 6379  # Redis
    
    # Enable firewall
    ufw --force enable
    
    success "Firewall configured"
}

# Deploy application
deploy_application() {
    info "Deploying CengBot application..."
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Install Python dependencies
    su - "$SERVICE_USER" -c "cd $PROJECT_DIR && /home/ceng/miniconda3/bin/pip install -r config/requirements.txt"
    
    # Initialize database
    su - "$SERVICE_USER" -c "cd $PROJECT_DIR && /home/ceng/miniconda3/bin/python -c 'from src.database_models import init_db; init_db()'"
    
    # Build frontend
    cd "$PROJECT_DIR/admin_frontend"
    su - "$SERVICE_USER" -c "cd $PROJECT_DIR/admin_frontend && npm install && npm run build"
    
    # Set proper permissions
    chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR"
    chmod -R 755 "$PROJECT_DIR"
    chmod +x "$PROJECT_DIR/scripts"/*.sh
    
    success "Application deployed"
}

# Apply Linux optimizations
apply_optimizations() {
    info "Applying Linux optimizations..."
    
    # Run optimization script
    if [[ -f "$PROJECT_DIR/scripts/optimize_linux.sh" ]]; then
        bash "$PROJECT_DIR/scripts/optimize_linux.sh"
    else
        warning "Optimization script not found, skipping"
    fi
    
    success "Linux optimizations applied"
}

# Start services
start_services() {
    info "Starting CengBot services..."
    
    # Enable and start services
    local services=("cengbot-worker" "cengbot-bot" "cengbot-admin" "cengbot-monitor")
    
    for service in "${services[@]}"; do
        systemctl daemon-reload
        systemctl enable "$service"
        systemctl start "$service"
        
        # Wait for service to start
        sleep 3
        
        if systemctl is-active --quiet "$service"; then
            success "$service started successfully"
        else
            error "Failed to start $service"
        fi
    done
    
    success "All services started"
}

# Post-deployment validation
validate_deployment() {
    info "Validating deployment..."
    
    # Check services
    local services=("cengbot-worker" "cengbot-bot" "cengbot-admin" "cengbot-monitor" "nginx" "rabbitmq-server")
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            success "$service is running"
        else
            error "$service is not running"
        fi
    done
    
    # Check API health
    sleep 5
    if curl -s http://localhost:8001/health > /dev/null; then
        success "Admin API is responding"
    else
        error "Admin API is not responding"
    fi
    
    # Check log files
    if [[ -f "$PROJECT_DIR/logs/worker.log" ]]; then
        success "Worker logs are being written"
    else
        warning "Worker logs not found"
    fi
    
    success "Deployment validation completed"
}

# Main deployment function
main() {
    info "Starting production deployment of CengBot..."
    
    # Check prerequisites
    check_root
    
    # Create backup
    create_backup
    
    # Stop existing services
    stop_services
    
    # Install dependencies
    install_dependencies
    
    # Setup infrastructure
    setup_nginx
    setup_log_rotation
    setup_monitoring
    setup_firewall
    
    # Deploy application
    deploy_application
    
    # Apply optimizations
    apply_optimizations
    
    # Start services
    start_services
    
    # Validate deployment
    validate_deployment
    
    success "Production deployment completed successfully!"
    
    # Show deployment summary
    info "Deployment Summary:"
    info "- CengBot services deployed and running"
    info "- Nginx reverse proxy configured"
    info "- Log rotation configured"
    info "- System monitoring enabled"
    info "- Firewall configured"
    info "- Linux optimizations applied"
    info ""
    info "Access URLs:"
    info "- Frontend: http://localhost/"
    info "- Admin API: http://localhost/api/"
    info "- Health Check: http://localhost/health"
    info ""
    info "Service Management:"
    info "- systemctl status cengbot-worker"
    info "- systemctl status cengbot-bot"
    info "- systemctl status cengbot-admin"
    info "- systemctl status cengbot-monitor"
    info ""
    info "Logs:"
    info "- tail -f $PROJECT_DIR/logs/worker.log"
    info "- tail -f $PROJECT_DIR/logs/bot.log"
    info "- tail -f $PROJECT_DIR/logs/admin.log"
    info "- tail -f $PROJECT_DIR/logs/monitor.log"
    
    warning "System restart recommended for all optimizations to take effect"
}

# Run main function
main "$@"
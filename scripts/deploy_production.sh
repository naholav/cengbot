#!/usr/bin/env bash
# =============================================================================
# Production deployment of CengBot (systemd + nginx). SYSTEM-WIDE, needs root.
#
# Usage: scripts/deploy_production.sh [--dry-run] [--yes] [--with-firewall] [--tune-system]
#
#   1. backup      project archive (code, .env, data, DB, models; NOT logs/.venv/
#                  node_modules/model_cache) -> <parent>/cengbot_deploy_backups/  (mode 600)
#   2. stop        systemd units cengbot-* and processes started by start_system.sh
#   3. packages    apt: nginx rabbitmq-server logrotate curl
#   4. app         pip install -r config/requirements.txt into .venv, init DB (missing
#                  tables only), npm install + build, frontend copied to /var/www/cengbot,
#                  .env -> mode 600
#   5. units       systemd units + logrotate via optimize_linux.sh --only services,logrotate
#                  (--tune-system: all optimize_linux.sh steps)
#   6. nginx       /etc/nginx/sites-available/cengbot (static panel, /api -> :8001)
#   7. monitoring  /usr/local/bin/cengbot-health.sh every 5 minutes (cron)
#   8. firewall    only with --with-firewall: allow ssh/80/443, enable ufw (no reset)
#   9. start       enable + start the units, check /health
#
# Shows this plan and asks before doing anything (--yes skips the question).
# Re-runs itself with sudo when needed (interactive password, never embedded).
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PROJECT_DIR="$PROJECT_ROOT"
SERVICE_USER="${CENGBOT_SERVICE_USER:-$(stat -c %U -- "$PROJECT_DIR")}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER" 2>/dev/null || echo "$SERVICE_USER")"
BACKUP_ROOT="${CENGBOT_DEPLOY_BACKUP_DIR:-$(dirname "$PROJECT_DIR")/cengbot_deploy_backups}"
WEB_ROOT="/var/www/cengbot"
UNITS=(cengbot-worker cengbot-bot cengbot-admin cengbot-monitor)
DRY_RUN=0; WITH_FIREWALL=0; TUNE_SYSTEM=0
ORIG_ARGS=("$@")

while (( $# > 0 )); do
    case "$1" in
        -n|--dry-run)    DRY_RUN=1 ;;
        -y|--yes)        ASSUME_YES=1 ;;
        --with-firewall) WITH_FIREWALL=1 ;;
        --tune-system)   TUNE_SYSTEM=1 ;;
        -h|--help)       sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               die "Unknown option: $1 (see --help)" ;;
    esac
    shift
done
export ASSUME_YES="${ASSUME_YES:-0}"

LOG_FILE="$PROJECT_DIR/logs/deployment.log"
log() {
    (( DRY_RUN == 1 )) && return 0
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE" 2>/dev/null || true
}
info()    { log "INFO: $1";    echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { log "SUCCESS: $1"; echo -e "${GREEN}✅ $1${NC}"; }
warning() { log "WARNING: $1"; echo -e "${YELLOW}⚠️  $1${NC}"; }
error()   { log "ERROR: $1";   echo -e "${RED}❌ $1${NC}" >&2; exit 1; }

run() {
    if (( DRY_RUN == 1 )); then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

# as_service_user 'shell commands' - login shell of the service user, with nvm if present
as_service_user() {
    local script="$1"
    if (( DRY_RUN == 1 )); then
        echo "  [dry-run] as $SERVICE_USER: $script"
        return 0
    fi
    su - "$SERVICE_USER" -s /bin/bash -c "export NVM_DIR=\"\$HOME/.nvm\"; [ -s \"\$NVM_DIR/nvm.sh\" ] && . \"\$NVM_DIR/nvm.sh\"; cd $(printf '%q' "$PROJECT_DIR") && $script"
}

write_root_file() {   # write_root_file PATH MODE  (content on stdin)
    local path="$1" mode="$2" tmp
    if (( DRY_RUN == 1 )); then
        echo "  [dry-run] would write $path"
        cat > /dev/null
        return 0
    fi
    if [[ -f "$path" ]]; then
        cp -p -- "$path" "$path.bak.$(date +%Y%m%d_%H%M%S)"
    fi
    tmp=$(mktemp "$path.XXXXXX")
    cat > "$tmp"
    chmod "$mode" "$tmp"
    mv -f -- "$tmp" "$path"
}

create_backup() {
    info "Backing up the current deployment to $BACKUP_ROOT"
    local archive
    archive="$BACKUP_ROOT/cengbot_$(date +%Y%m%d_%H%M%S).tar.gz"
    run mkdir -p "$BACKUP_ROOT"
    run chmod 700 "$BACKUP_ROOT"
    # Contains .env (secrets) -> archive is readable by root only.
    run tar -C "$(dirname "$PROJECT_DIR")" -czf "$archive" \
        --exclude="$(basename "$PROJECT_DIR")/logs" \
        --exclude="$(basename "$PROJECT_DIR")/.venv" \
        --exclude="$(basename "$PROJECT_DIR")/model_cache" \
        --exclude="$(basename "$PROJECT_DIR")/backups" \
        --exclude="$(basename "$PROJECT_DIR")/checkpoints" \
        --exclude="$(basename "$PROJECT_DIR")/admin_frontend/node_modules" \
        --exclude="$(basename "$PROJECT_DIR")/admin_frontend/build" \
        --exclude='__pycache__' \
        "$(basename "$PROJECT_DIR")"
    run chmod 600 "$archive"
    for unit in "${UNITS[@]}"; do
        if [[ -f "/etc/systemd/system/$unit.service" ]]; then
            run cp -p "/etc/systemd/system/$unit.service" "$BACKUP_ROOT/"
        fi
    done
    success "Backup: $archive (older backups are kept - delete them manually when no longer needed)"
}

stop_services() {
    info "Stopping running CengBot services..."
    for unit in "${UNITS[@]}"; do
        if systemctl is-active --quiet "$unit"; then
            run systemctl stop "$unit"
            info "Stopped $unit"
        fi
    done
    as_service_user "scripts/stop_system.sh >/dev/null 2>&1 || true"
    success "Services stopped"
}

install_dependencies() {
    info "Installing system packages..."
    run apt-get update
    run apt-get install -y nginx rabbitmq-server logrotate curl
    run systemctl enable --now rabbitmq-server
    success "System packages installed"
}

deploy_application() {
    info "Deploying the application as $SERVICE_USER..."
    if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
        error "$PROJECT_DIR/.venv/bin/python missing - create the venv first (scripts/setup_system.sh)"
    fi
    as_service_user ".venv/bin/python -m pip install -r config/requirements.txt"
    as_service_user ".venv/bin/python -c 'import sys; sys.path.insert(0, \"src\"); from database_models import init_db; init_db()'"
    # The panel is served by nginx on the same origin; nginx proxies /api/ to the
    # admin API (127.0.0.1:8001). REACT_APP_API_URL is baked in at build time.
    as_service_user "cd admin_frontend && npm install && REACT_APP_API_URL=$(printf '%q' "${PANEL_API_URL:-/api}") npm run build"
    run install -d -m 0755 "$WEB_ROOT"
    # Remove files of the previous build (e.g. old static/ chunks) before copying.
    if [[ -n "$WEB_ROOT" && "$WEB_ROOT" == /var/www/* && -d "$WEB_ROOT" ]]; then
        run find "$WEB_ROOT" -mindepth 1 -delete
    fi
    run cp -a "$PROJECT_DIR/admin_frontend/build/." "$WEB_ROOT/"
    run chmod -R a+rX "$WEB_ROOT"
    # Secrets must not be readable by other users (the old version did chmod -R 755).
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        run chown "$SERVICE_USER:$SERVICE_GROUP" "$PROJECT_DIR/.env"
        run chmod 600 "$PROJECT_DIR/.env"
    fi
    run chmod +x "$PROJECT_DIR"/scripts/*.sh
    success "Application deployed"
}

install_units() {
    local steps="services,logrotate"
    (( TUNE_SYSTEM == 1 )) && steps="kernel,limits,gpu,cpu,services,logrotate"
    info "systemd units / log rotation (optimize_linux.sh --only $steps)"
    if (( DRY_RUN == 1 )); then
        "$SCRIPTS_DIR/optimize_linux.sh" --dry-run --only "$steps"
    else
        CENGBOT_SERVICE_USER="$SERVICE_USER" "$SCRIPTS_DIR/optimize_linux.sh" --yes --only "$steps"
    fi
}

setup_nginx() {
    info "Configuring nginx..."
    write_root_file /etc/nginx/sites-available/cengbot 0644 <<EOF
# CengBot - managed by scripts/deploy_production.sh
server {
    listen 80;
    server_name _;

    # Admin panel (static React build copied to $WEB_ROOT)
    root $WEB_ROOT;
    index index.html;
    location / {
        try_files \$uri /index.html;
    }

    # Admin API (CORS is handled by the API itself - no wildcard here)
    location /api/ {
        proxy_pass http://127.0.0.1:8001/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /health {
        proxy_pass http://127.0.0.1:8001/health;
        access_log off;
    }

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript;
}
EOF
    run ln -sf /etc/nginx/sites-available/cengbot /etc/nginx/sites-enabled/cengbot
    if [[ -L /etc/nginx/sites-enabled/default ]]; then
        run rm -f /etc/nginx/sites-enabled/default   # only the symlink; sites-available/default stays
    fi
    run nginx -t
    run systemctl enable nginx
    run systemctl reload-or-restart nginx
    success "nginx configured"
}

setup_monitoring() {
    info "Installing the 5-minute health check..."
    write_root_file /usr/local/bin/cengbot-health.sh 0755 <<'EOF'
#!/bin/bash
# CengBot health monitoring - managed by scripts/deploy_production.sh
for service in cengbot-worker cengbot-bot cengbot-admin rabbitmq-server nginx; do
    if ! systemctl is-active --quiet "$service"; then
        echo "$(date '+%F %T') ERROR: $service is not running - starting it"
        systemctl start "$service"
    fi
done
disk_usage=$(df -P / | awk 'NR==2 {gsub("%", "", $5); print $5}')
if [[ "$disk_usage" -gt 90 ]]; then
    echo "$(date '+%F %T') WARNING: disk usage is ${disk_usage}%"
fi
if ! curl -s -f -m 10 http://localhost:8001/health > /dev/null; then
    echo "$(date '+%F %T') ERROR: Admin API is not responding - restarting cengbot-admin"
    systemctl restart cengbot-admin
fi
EOF
    write_root_file /etc/cron.d/cengbot-health 0644 <<'EOF'
# CengBot health monitoring - managed by scripts/deploy_production.sh
*/5 * * * * root /usr/local/bin/cengbot-health.sh >> /var/log/cengbot-health.log 2>&1
EOF
    write_root_file /etc/logrotate.d/cengbot-health 0644 <<'EOF'
/var/log/cengbot-health.log {
    size 10M
    rotate 3
    copytruncate
    compress
    missingok
    notifempty
}
EOF
    success "Health monitoring configured"
}

setup_firewall() {
    if (( WITH_FIREWALL == 0 )); then
        info "Firewall: skipped (pass --with-firewall to configure ufw)"
        return 0
    fi
    info "Configuring ufw (existing rules are kept)..."
    run apt-get install -y ufw
    run ufw allow OpenSSH
    run ufw allow 80/tcp
    run ufw allow 443/tcp
    run ufw default deny incoming
    run ufw default allow outgoing
    run ufw --force enable
    success "Firewall configured"
}

start_services() {
    info "Starting CengBot services..."
    run systemctl daemon-reload
    for unit in "${UNITS[@]}"; do
        run systemctl enable "$unit"
        run systemctl restart "$unit"
    done
    (( DRY_RUN == 1 )) && return 0
    sleep 5
    for unit in "${UNITS[@]}" nginx rabbitmq-server; do
        if systemctl is-active --quiet "$unit"; then
            success "$unit is running"
        else
            error "$unit is not running (journalctl -u $unit, logs/ for output)"
        fi
    done
    local ok=0
    for _ in $(seq 1 12); do
        if curl -s -f -m 5 http://localhost:8001/health > /dev/null; then ok=1; break; fi
        sleep 5
    done
    if (( ok == 1 )); then success "Admin API is responding"; else warning "Admin API /health not responding yet"; fi
    info "The AI worker needs a few minutes to load the model: tail -f $PROJECT_DIR/logs/worker.log"
}

# Fail BEFORE anything is stopped if the panel cannot be built: Vite needs
# Node.js ^22.13 or >=24 for the service user (nvm is loaded by as_service_user).
preflight_build_tools() {
    info "Checking build prerequisites for $SERVICE_USER..."
    if ! as_service_user "node -e 'const [a, b] = process.versions.node.split(\".\").map(Number); process.exit(a >= 24 || (a === 22 && b >= 13) ? 0 : 1)' && command -v npm >/dev/null"; then
        error "Node.js 22.13+ (or 24+) with npm is required for $SERVICE_USER to build the admin panel. $NODE_INSTALL_HINT"
    fi
}

main() {
    print_header "CengBot production deployment"
    sed -n '6,20p' "$0" | sed 's/^# \{0,1\}//'
    echo "Project: $PROJECT_DIR   service user: $SERVICE_USER   firewall: $( (( WITH_FIREWALL )) && echo yes || echo no )   full tuning: $( (( TUNE_SYSTEM )) && echo yes || echo no )"
    echo
    if (( DRY_RUN == 1 )); then
        print_status "Dry run - nothing will be changed"
    else
        reexec_as_root "${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}"
        confirm "Run the production deployment with the plan above?" || die "Cancelled - nothing was changed."
        mkdir -p "$PROJECT_DIR/logs"
        touch "$LOG_FILE" && chown "$SERVICE_USER:$SERVICE_GROUP" "$LOG_FILE" 2>/dev/null || true
    fi
    preflight_build_tools
    create_backup
    stop_services
    install_dependencies
    deploy_application
    install_units
    setup_nginx
    setup_monitoring
    setup_firewall
    start_services
    success "Production deployment completed"
    info "Status: systemctl status ${UNITS[*]}"
    info "Logs:   $PROJECT_DIR/logs/{worker,bot,admin,monitor}.log (rotated by /etc/logrotate.d/cengbot)"
    info "Panel:  http://<server>/   API: http://<server>/api/   Health: http://<server>/health"
}

main

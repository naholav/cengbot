#!/usr/bin/env bash
# =============================================================================
# CengBot - start the complete system (manual / non-systemd mode)
#   AI worker -> Admin API -> Telegram bot -> React admin panel
#
# Options:
#   --install-deps   (re)install Python requirements and frontend npm packages
#   --no-frontend    do not start the admin panel dev server (Vite)
#   -h, --help
#
# Logs are APPENDED to logs/*.log; any log bigger than CENGBOT_LOG_MAX_BYTES
# (default 50 MiB) is rotated first (last 10 MiB kept in <log>.1.gz, 5 generations).
# =============================================================================
set -uo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

INSTALL_DEPS=0
START_FRONTEND=1
while (( $# > 0 )); do
    case "$1" in
        --install-deps) INSTALL_DEPS=1 ;;
        --no-frontend)  START_FRONTEND=0 ;;
        -h|--help)      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              die "Unknown option: $1 (see --help)" ;;
    esac
    shift
done

cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
echo "🚀 CengBot Complete System Starting..."
echo "====================================="

if [[ ! -f "README.md" || ! -d "src" ]]; then
    die "$PROJECT_ROOT does not look like the CengBot project (README.md / src missing)"
fi

# Refuse to run a second copy next to the systemd-managed services.
for unit in cengbot-worker cengbot-bot cengbot-admin; do
    if systemd_unit_active_here "$unit"; then
        die "$unit.service is active (systemd mode). Use 'sudo systemctl restart $unit' instead of this script."
    fi
done

# ---- requirements -------------------------------------------------------------
print_status "Checking requirements..."
install_pkg() {   # install_pkg <apt packages...>
    print_warning "Installing: $* (sudo will ask for YOUR password)"
    as_root apt-get update && as_root apt-get install -y "$@"
}
if ! command -v rabbitmq-server >/dev/null 2>&1 && ! systemctl cat rabbitmq-server.service >/dev/null 2>&1; then
    print_warning "RabbitMQ is not installed"
    install_pkg rabbitmq-server || die "RabbitMQ installation failed"
fi
if (( START_FRONTEND == 1 )) && ! command -v npm >/dev/null 2>&1; then
    print_warning "Node.js / npm not found"
    install_pkg nodejs npm || die "Node.js installation failed"
fi

VENV_CREATED=0
if [[ ! -x "$PYTHON" ]]; then
    if [[ -n "${CENGBOT_PYTHON:-}" ]]; then
        die "CENGBOT_PYTHON=$CENGBOT_PYTHON is not executable"
    fi
    BASE_PY="${CENGBOT_BASE_PYTHON:-python3}"
    command -v "$BASE_PY" >/dev/null 2>&1 || die "$BASE_PY not found - install Python 3.11 first"
    print_status "Creating Python virtual environment .venv with $BASE_PY ..."
    "$BASE_PY" -m venv .venv || die "venv creation failed"
    VENV_CREATED=1
fi
require_python
print_success "Python: $("$PYTHON" --version 2>&1) ($PYTHON)"

if (( INSTALL_DEPS == 1 || VENV_CREATED == 1 )); then
    print_status "Installing Python dependencies..."
    "$PYTHON" -m pip install -r config/requirements.txt || die "pip install failed"
fi
# no .bin/vite: node_modules is missing or still holds the old Create React App install
if (( START_FRONTEND == 1 )) && { (( INSTALL_DEPS == 1 )) || [[ ! -x admin_frontend/node_modules/.bin/vite ]]; }; then
    print_status "Installing frontend dependencies..."
    ( cd admin_frontend && npm install ) || die "npm install failed"
fi

print_status "Initializing database (creates missing tables only)..."
( cd "$PROJECT_ROOT" && "$PYTHON" -c 'import sys; sys.path.insert(0, "src"); from database_models import init_db; init_db()' ) \
    || die "Database initialization failed"

# ---- RabbitMQ -------------------------------------------------------------------
if ! systemctl is-active --quiet rabbitmq-server; then
    print_status "Starting RabbitMQ (sudo may ask for YOUR password)..."
    as_root systemctl start rabbitmq-server || die "Could not start rabbitmq-server"
fi
if ! systemctl is-enabled --quiet rabbitmq-server 2>/dev/null; then
    as_root systemctl enable rabbitmq-server || print_warning "Could not enable rabbitmq-server at boot"
fi
for _ in $(seq 1 30); do
    systemctl is-active --quiet rabbitmq-server && break
    sleep 1
done
systemctl is-active --quiet rabbitmq-server || die "RabbitMQ failed to start (journalctl -u rabbitmq-server)"
print_success "RabbitMQ is running"

# ---- stop old instances of THIS project (never other processes) --------------
mkdir -p "$LOG_DIR"
print_status "Stopping existing CengBot services of this project..."
for svc in "${CENGBOT_SERVICES[@]}" legacy_bot; do
    mapfile -t pids < <(service_pids "$svc")
    if (( ${#pids[@]} > 0 )); then
        stop_pids "$(service_label "$svc")" 30 "${pids[@]}"
    fi
done

print_status "Rotating oversized logs..."
rotate_logs_dir "$LOG_DIR"

# ---- start ----------------------------------------------------------------------
echo ""
print_status "Starting services..."
echo "=============================="

declare -A PIDS=([worker]="-" [api]="-" [bot]="-" [frontend]="-")
start_and_record() {
    start_service "$1"
    PIDS[$1]="$LAST_PID"
    pidfile_write "${PIDS[worker]}" "${PIDS[api]}" "${PIDS[bot]}" "${PIDS[frontend]}"
    print_success "$(service_label "$1") started - PID: $LAST_PID (log: $(service_log "$1"))"
}

print_status "1️⃣  AI Worker (LLaMA 3.2 3B + LoRA from models/active-model)..."
start_and_record worker
sleep 5
print_status "2️⃣  Admin API..."
start_and_record api
sleep 3
print_status "3️⃣  Telegram Bot..."
start_and_record bot
if (( START_FRONTEND == 1 )); then
    sleep 3
    print_status "4️⃣  React Frontend..."
    start_and_record frontend
fi

echo ""
print_status "⏳ Waiting for services to initialize..."
sleep 10

echo ""
print_status "🔍 Service status:"
FAILED=0
for svc in "${CENGBOT_SERVICES[@]}"; do
    pid="${PIDS[$svc]}"
    [[ "$pid" == "-" ]] && continue
    if pid_alive "$pid"; then
        print_success "$(service_label "$svc") is running (PID $pid)"
    else
        print_error "$(service_label "$svc") failed to start - check $(service_log "$svc")"
        FAILED=1
    fi
done

echo ""
echo "🌐 Admin Panel: http://localhost:3000    API docs: http://localhost:8001/docs"
echo "📝 Logs: tail -f logs/worker.log | logs/bot.log | logs/admin.log | logs/frontend.log"
echo "🧠 Active model: $(readlink models/active-model 2>/dev/null || echo 'NOT SET') (scripts/switch_model.sh current)"
echo "🛑 Stop: ./scripts/stop_system.sh"
echo "💡 First model loading may take several minutes."
exit "$FAILED"

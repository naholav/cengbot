#!/usr/bin/env bash
# =============================================================================
# CengBot system setup (Ubuntu 22.04+)
#
# Usage: scripts/setup_system.sh [--upgrade] [--yes]
#   --upgrade   also run "apt upgrade" (off by default: a full upgrade can replace
#               the NVIDIA driver / CUDA stack the model needs)
#   --yes       do not ask for confirmation
#
# Installs system packages (sudo asks for YOUR password), creates .venv only if
# it does not exist yet (an existing venv is never re-created), installs the
# Python and frontend dependencies, initialises the database (missing tables
# only) and creates .env from .env.example if .env is missing.
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

UPGRADE=0
while (( $# > 0 )); do
    case "$1" in
        --upgrade) UPGRADE=1 ;;
        -y|--yes)  ASSUME_YES=1 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         die "Unknown option: $1 (see --help)" ;;
    esac
    shift
done
export ASSUME_YES="${ASSUME_YES:-0}"

cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
echo "🔧 CengBot System Setup ($PROJECT_ROOT)"
echo "======================"
confirm "Install system packages (sudo) and project dependencies?" || die "Cancelled."

print_status "📦 Installing system packages..."
as_root apt-get update
if (( UPGRADE == 1 )); then
    as_root apt-get upgrade -y
fi
as_root apt-get install -y python3 python3-pip python3-venv python3-dev \
    rabbitmq-server logrotate curl wget git htop dos2unix build-essential
# Node.js is not taken from apt: distribution packages are too old for Vite.
if node_ok; then
    print_success "Node.js $(node --version) OK for the admin panel"
else
    print_warning "Node.js is missing or too old for the admin panel. $NODE_INSTALL_HINT"
fi

print_status "🔄 Configuring RabbitMQ..."
as_root systemctl enable --now rabbitmq-server
as_root rabbitmq-plugins enable rabbitmq_management || print_warning "Could not enable the RabbitMQ management plugin"

if [[ -x "$PYTHON" ]]; then
    print_success "Using existing Python environment: $PYTHON ($("$PYTHON" --version 2>&1))"
else
    BASE_PY="${CENGBOT_BASE_PYTHON:-python3}"
    print_status "🐍 Creating .venv with $BASE_PY (set CENGBOT_BASE_PYTHON to use e.g. a conda python with CUDA torch)"
    "$BASE_PY" -m venv .venv
    "$PYTHON" -m pip install --upgrade pip setuptools wheel
fi

print_status "📦 Installing Python dependencies..."
"$PYTHON" -m pip install -r config/requirements.txt

if node_ok && command -v npm >/dev/null 2>&1; then
    print_status "📦 Installing Node.js dependencies..."
    ( cd admin_frontend && npm install )
else
    print_warning "Skipping the admin panel dependencies (Node.js 22.13+/24+ with npm is required). $NODE_INSTALL_HINT"
fi

print_status "💾 Initializing database (creates missing tables only)..."
"$PYTHON" -c 'import sys; sys.path.insert(0, "src"); from database_models import init_db; init_db()'

mkdir -p logs config/env
find scripts/ -maxdepth 1 -name "*.sh" -exec chmod +x {} +

if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        chmod 600 .env
        print_warning "Created .env from .env.example - edit it with your real values (nano .env)"
    else
        print_warning ".env.example not found - create .env manually"
    fi
else
    chmod 600 .env 2>/dev/null || true
fi

echo ""
print_success "🎉 CengBot setup complete"
echo "Next steps:"
echo "  1. Edit .env (TELEGRAM_BOT_TOKEN, ADMIN_PASSWORD_HASH, ...)"
echo "  2. Start:         ./scripts/start_system.sh   (or systemd: sudo ./scripts/optimize_linux.sh --only services,logrotate)"
echo "  3. Log rotation:  ./scripts/rotate_logs.sh --install-cron   (no root)"
echo "  4. Check:         ./scripts/health_check.sh"
echo "Service URLs: Admin Panel http://localhost:3000 | API docs http://localhost:8001/docs | RabbitMQ http://localhost:15672"

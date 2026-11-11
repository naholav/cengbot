#!/usr/bin/env bash
# =============================================================================
# CengBot system health check (read-only: changes nothing)
# =============================================================================
set -uo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ok()   { printf '%s✅ %s%s\n' "$GREEN" "$*" "$NC"; }
warn() { printf '%s⚠️  %s%s\n' "$YELLOW" "$*" "$NC"; }
bad()  { printf '%s❌ %s%s\n' "$RED" "$*" "$NC"; }

cd "$PROJECT_ROOT" || exit 1
echo "🔍 CengBot System Health Check"
echo "=============================="

echo -e "\n${BLUE}🔍 Dependencies${NC}"
if [[ -x "$PYTHON" ]]; then
    ok "Python: $("$PYTHON" --version 2>&1 | cut -d' ' -f2) ($PYTHON)"
else
    bad "Project Python not found: $PYTHON"
fi
# the admin panel toolchain (Vite, Vitest) needs Node.js ^22.13 or >=24 (admin_frontend/package.json engines)
if ! command -v node >/dev/null 2>&1; then
    bad "Node.js not found"
elif node -e 'const [a, b] = process.versions.node.split(".").map(Number); process.exit(a >= 24 || (a === 22 && b >= 13) ? 0 : 1)' 2>/dev/null; then
    ok "Node.js: $(node --version 2>&1)"
else
    warn "Node.js: $(node --version 2>&1) - too old for the admin panel (needs 22.13+ or 24+)"
fi
if systemctl cat rabbitmq-server.service >/dev/null 2>&1 || command -v rabbitmq-server >/dev/null 2>&1; then
    if systemctl is-active --quiet rabbitmq-server; then ok "RabbitMQ: running"; else warn "RabbitMQ: installed but not running"; fi
else
    bad "RabbitMQ not found"
fi

echo -e "\n${BLUE}🔍 CengBot services${NC}"
RUNNING=0
for svc in "${CENGBOT_SERVICES[@]}"; do
    pids=$(service_pids "$svc" | tr '\n' ' ')
    unit=""
    case "$svc" in worker) unit=cengbot-worker ;; api) unit=cengbot-admin ;; bot) unit=cengbot-bot ;; esac
    if [[ -n "${pids// /}" ]]; then
        ok "$(service_label "$svc"): running (PID: ${pids% })"
        RUNNING=$(( RUNNING + 1 ))
    elif [[ -n "$unit" ]] && systemd_unit_active_here "$unit"; then
        ok "$(service_label "$svc"): running ($unit.service)"
        RUNNING=$(( RUNNING + 1 ))
    else
        bad "$(service_label "$svc"): not running"
    fi
done
legacy=$(service_pids legacy_bot | tr '\n' ' ')
if [[ -n "${legacy// /}" ]]; then
    warn "$(service_label legacy_bot) is running (PID: ${legacy% }) - it conflicts with the production bot; stop it"
fi
if command -v curl >/dev/null 2>&1; then
    if curl -s -f -m 5 http://localhost:8001/health > /dev/null 2>&1; then ok "Admin API /health: OK"; else warn "Admin API /health: not responding"; fi
    if curl -s -f -m 5 http://localhost:3000 > /dev/null 2>&1; then ok "Frontend: responding"; else warn "Frontend: not responding"; fi
fi

echo -e "\n${BLUE}🧠 Active model${NC}"
if [[ -L models/active-model ]]; then
    if "$SCRIPTS_DIR/switch_model.sh" validate "$PROJECT_ROOT/models/active-model" >/dev/null 2>&1; then
        ok "models/active-model -> $(readlink models/active-model) (valid)"
    else
        bad "models/active-model -> $(readlink models/active-model) is NOT a valid model (scripts/switch_model.sh current)"
    fi
else
    bad "models/active-model symlink missing (scripts/switch_model.sh list)"
fi

echo -e "\n${BLUE}🔍 Resources${NC}"
DISK_USAGE=$(df -P . | awk 'NR==2 {gsub("%", "", $5); print $5}')
if [[ "$DISK_USAGE" =~ ^[0-9]+$ ]] && (( DISK_USAGE < 80 )); then
    ok "Disk usage: ${DISK_USAGE}%"
else
    warn "Disk usage: ${DISK_USAGE}% (high)"
fi
MEM_USAGE=$(free | awk '/^Mem/ {printf "%.1f", $3/$2 * 100.0}')
ok "Memory usage: ${MEM_USAGE}%"
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -n 1)"
fi
DB_FILE="$(resolve_db_file)"
if [[ -f "$DB_FILE" ]]; then ok "Database: $DB_FILE ($(du -h "$DB_FILE" | cut -f1))"; else bad "Database: not found ($DB_FILE)"; fi

echo -e "\n${BLUE}🔍 Log files${NC}"
for logfile in logs/*.log; do
    [[ -f "$logfile" ]] || continue
    size=$(stat -c %s -- "$logfile")
    # Never read big logs completely (line counting a multi-GB file takes minutes).
    if (( size > CENGBOT_LOG_MAX_BYTES )); then
        warn "$(basename "$logfile"): $(human_size "$size") - larger than $(human_size "$CENGBOT_LOG_MAX_BYTES"), rotated on next start"
    else
        ok "$(basename "$logfile"): $(human_size "$size") ($(wc -l < "$logfile") lines)"
    fi
done

echo -e "\n${BLUE}🔍 Ports${NC}"
if command -v ss >/dev/null 2>&1; then
    for spec in "8001:Admin API" "3000:Frontend" "5672:RabbitMQ"; do
        port="${spec%%:*}"
        if ss -tuln | grep -q ":${port} "; then ok "Port $port: open (${spec#*:})"; else bad "Port $port: closed (${spec#*:})"; fi
    done
fi

echo -e "\n${BLUE}📊 Summary${NC}"
if (( RUNNING == 4 )); then
    ok "All services are running (4/4)"
elif (( RUNNING > 0 )); then
    warn "Partial system running ($RUNNING/4 services) - try ./scripts/start_system.sh"
else
    bad "System is not running (0/4 services) - start with ./scripts/start_system.sh"
fi
echo ""

#!/usr/bin/env bash
# =============================================================================
# Linux tuning, systemd units and log rotation for CengBot (system-wide, root)
#
# Usage: scripts/optimize_linux.sh [--dry-run] [--yes] [--only step1,step2]
# Steps (default: all, in this order):
#   kernel     /etc/sysctl.d/99-cengbot-optimization.conf (swappiness, dirty ratios,
#              network buffers, BBR if the kernel supports it)
#   limits     /etc/security/limits.d/99-cengbot.conf (open files for the service user)
#   gpu        NVIDIA persistence mode (nvidia-persistenced)
#   cpu        CPU governor "performance" (+ oneshot unit to keep it after reboot)
#   services   cengbot-worker / -bot / -admin / -monitor systemd units (not started)
#   logrotate  /etc/logrotate.d/cengbot (config/logrotate/cengbot.conf) + hourly cron
#
# Prints the plan and asks before changing anything (--yes to skip, --dry-run to
# only print). Re-runs itself with sudo when needed (interactive password).
#
# Removed from the earlier version on purpose (harmful or broken):
#   fs.file-max=65536 (LOWERED the kernel default), vm.overcommit_memory=1,
#   vm.nr_hugepages (reserved 2 GB, appended on every run), fixed GPU clocks
#   (-ac 877,1455 is invalid for this GPU), GPU power cap 300 W, ECC toggle,
#   disabling all CPU idle states, I/O scheduler/queue-depth overrides for NVMe,
#   writes to /proc/sys/kernel/sched_* (do not exist on kernel >= 5.13),
#   Nice=-10 / SCHED_FIFO for the services.
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PROJECT_DIR="$PROJECT_ROOT"
SYSTEMD_DIR="${CENGBOT_SYSTEMD_DIR:-/etc/systemd/system}"
SERVICE_USER="${CENGBOT_SERVICE_USER:-$(stat -c %U -- "$PROJECT_DIR")}"
SERVICE_GROUP="$(id -gn "$SERVICE_USER" 2>/dev/null || echo "$SERVICE_USER")"
VENV_PY="$PROJECT_DIR/.venv/bin/python"
ALL_STEPS=(kernel limits gpu cpu services logrotate)
STEPS=("${ALL_STEPS[@]}")
DRY_RUN=0
ORIG_ARGS=("$@")

while (( $# > 0 )); do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        -y|--yes)     ASSUME_YES=1 ;;
        --only)       shift; [[ $# -gt 0 ]] || die "--only needs a list"; IFS=',' read -r -a STEPS <<< "$1" ;;
        --only=*)     IFS=',' read -r -a STEPS <<< "${1#--only=}" ;;
        -h|--help)    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            die "Unknown option: $1 (see --help)" ;;
    esac
    shift
done
export ASSUME_YES="${ASSUME_YES:-0}"
for s in "${STEPS[@]}"; do
    [[ " ${ALL_STEPS[*]} " == *" $s "* ]] || die "Unknown step '$s' (valid: ${ALL_STEPS[*]})"
done

LOG_FILE="$PROJECT_DIR/logs/optimization.log"
log() {
    (( DRY_RUN == 1 )) && return 0
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE" 2>/dev/null || true
}
info()    { log "INFO: $1";    echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { log "SUCCESS: $1"; echo -e "${GREEN}✅ $1${NC}"; }
warning() { log "WARNING: $1"; echo -e "${YELLOW}⚠️  $1${NC}"; }

# run CMD... : execute, or only print in --dry-run
run() {
    if (( DRY_RUN == 1 )); then
        echo "  [dry-run] $*"
    else
        "$@"
    fi
}

# install_file PATH MODE  (content on stdin)
install_file() {
    local path="$1" mode="$2" tmp
    if (( DRY_RUN == 1 )); then
        echo "  [dry-run] would write $path (mode $mode):"
        sed 's/^/      | /'
        return 0
    fi
    tmp=$(mktemp "${path}.XXXXXX")
    cat > "$tmp"
    chmod "$mode" "$tmp"
    mv -f -- "$tmp" "$path"
    echo "  wrote $path"
}

step_kernel() {
    info "Kernel parameters"
    local bbr="# net.ipv4.tcp_congestion_control = bbr   (tcp_bbr not available)"
    if (( DRY_RUN == 1 )) || modprobe -q tcp_bbr 2>/dev/null; then
        if (( DRY_RUN == 1 )) || grep -qw bbr /proc/sys/net/ipv4/tcp_available_congestion_control 2>/dev/null; then
            bbr="net.ipv4.tcp_congestion_control = bbr"
        fi
    fi
    install_file /etc/sysctl.d/99-cengbot-optimization.conf 0644 <<EOF
# CengBot - managed by scripts/optimize_linux.sh (safe to delete)
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000
$bbr
EOF
    run sysctl -p /etc/sysctl.d/99-cengbot-optimization.conf || warning "some sysctl values could not be applied"
    success "Kernel parameters configured"
}

step_limits() {
    info "System limits for $SERVICE_USER"
    install_file /etc/security/limits.d/99-cengbot.conf 0644 <<EOF
# CengBot - managed by scripts/optimize_linux.sh (applies to new login sessions)
$SERVICE_USER soft nofile 65536
$SERVICE_USER hard nofile 65536
$SERVICE_USER soft memlock unlimited
$SERVICE_USER hard memlock unlimited
EOF
    success "System limits configured"
}

step_gpu() {
    info "GPU persistence mode"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warning "NVIDIA GPU not detected - skipped"
        return 0
    fi
    if systemctl cat nvidia-persistenced.service >/dev/null 2>&1; then
        run systemctl start nvidia-persistenced || warning "could not start nvidia-persistenced"
    else
        run nvidia-smi -pm 1 || warning "could not enable persistence mode"
    fi
    success "GPU persistence mode configured (clocks / power limit left at driver defaults)"
}

step_cpu() {
    info "CPU governor -> performance"
    if ! compgen -G "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor" >/dev/null; then
        warning "cpufreq not available - skipped"
        return 0
    fi
    install_file /usr/local/bin/cengbot-cpu-performance.sh 0755 <<'EOF'
#!/bin/sh
# CengBot - set the CPU frequency governor to "performance" (managed by optimize_linux.sh)
for g in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
    echo performance > "$g" 2>/dev/null || true
done
EOF
    install_file "$SYSTEMD_DIR/cengbot-cpu-performance.service" 0644 <<'EOF'
[Unit]
Description=CengBot: CPU governor performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/cengbot-cpu-performance.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    run systemctl daemon-reload
    run systemctl enable --now cengbot-cpu-performance.service || warning "could not enable cengbot-cpu-performance.service"
    success "CPU governor configured"
}

# unit NAME DESCRIPTION SCRIPT LOG [extra [Unit] lines] [extra [Service] lines]
write_unit() {
    local name="$1" desc="$2" script="$3" logname="$4" unit_extra="$5" service_extra="$6"
    install_file "$SYSTEMD_DIR/$name.service" 0644 <<EOF
[Unit]
Description=$desc
After=network-online.target rabbitmq-server.service
Wants=network-online.target
$unit_extra

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_GROUP
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONPATH=$PROJECT_DIR/src
Environment=PYTHONUNBUFFERED=1
ExecStart=$VENV_PY $script
Restart=always
RestartSec=10
# Appended (O_APPEND): rotated by logrotate with copytruncate (config/logrotate/cengbot.conf)
StandardOutput=append:$PROJECT_DIR/logs/$logname
StandardError=append:$PROJECT_DIR/logs/$logname
LimitNOFILE=65536
$service_extra

[Install]
WantedBy=multi-user.target
EOF
}

step_services() {
    info "CengBot systemd units (interpreter: $VENV_PY, user: $SERVICE_USER)"
    [[ -x "$VENV_PY" ]] || warning "$VENV_PY does not exist yet - create the venv before starting the services"
    run install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0775 "$PROJECT_DIR/logs"
    write_unit cengbot-worker "CengBot AI Worker" src/ai_model_worker.py worker.log \
        "Wants=rabbitmq-server.service" $'TimeoutStopSec=60\nLimitMEMLOCK=infinity'
    write_unit cengbot-bot "CengBot Telegram Bot" src/telegram_bot_rabbitmq.py bot.log \
        "Wants=rabbitmq-server.service" ""
    write_unit cengbot-admin "CengBot Admin API" src/admin_rest_api.py admin.log "" ""
    write_unit cengbot-monitor "CengBot System Monitor" src/system_monitor.py monitor.log "" ""
    run systemctl daemon-reload
    success "Units installed (enable/start: sudo systemctl enable --now cengbot-worker cengbot-bot cengbot-admin cengbot-monitor)"
    info "Do not run scripts/start_system.sh at the same time as these units."
}

step_logrotate() {
    info "Log rotation (size 50M, rotate 5, copytruncate, compress)"
    local lr
    lr=$(command -v logrotate || echo /usr/sbin/logrotate)
    if [[ ! -x "$lr" ]] && (( DRY_RUN == 0 )); then
        run apt-get install -y logrotate || { warning "logrotate not installed - skipped"; return 0; }
    fi
    CENGBOT_SERVICE_USER="$SERVICE_USER" "$SCRIPTS_DIR/rotate_logs.sh" --print-config --system \
        | install_file /etc/logrotate.d/cengbot 0644
    install_file /etc/cron.hourly/cengbot-logrotate 0755 <<EOF
#!/bin/sh
# CengBot - hourly size-based log rotation (managed by scripts/optimize_linux.sh)
exec $lr -s /var/lib/logrotate/cengbot.status /etc/logrotate.d/cengbot
EOF
    if (( DRY_RUN == 0 )); then
        if "$lr" -d -s /var/lib/logrotate/cengbot.status /etc/logrotate.d/cengbot >/dev/null 2>&1; then
            success "Log rotation installed (/etc/logrotate.d/cengbot, hourly: /etc/cron.hourly/cengbot-logrotate)"
        else
            warning "logrotate -d reported a problem: sudo logrotate -d /etc/logrotate.d/cengbot"
        fi
    fi
}

main() {
    print_header "CengBot Linux optimization"
    echo "Project:      $PROJECT_DIR"
    echo "Service user: $SERVICE_USER:$SERVICE_GROUP"
    echo "Steps:        ${STEPS[*]}"
    if (( DRY_RUN == 1 )); then
        echo "Mode:         dry run - nothing will be changed"
    else
        reexec_as_root "${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"}"
        confirm "Apply these SYSTEM-WIDE changes?" || die "Cancelled - nothing was changed."
        mkdir -p "$PROJECT_DIR/logs"
        touch "$LOG_FILE" && chown "$SERVICE_USER:$SERVICE_GROUP" "$LOG_FILE" 2>/dev/null || true
    fi
    local s
    for s in "${STEPS[@]}"; do
        "step_$s"
    done
    success "Done. Changes are listed in $LOG_FILE"
}

main

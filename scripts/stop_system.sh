#!/usr/bin/env bash
# =============================================================================
# CengBot - stop the services started by scripts/start_system.sh
#
# Only processes that belong to THIS project are stopped: PIDs from .pids are
# verified against their command line (protects against PID reuse), and the
# fallback search only matches CengBot scripts running inside the project
# directory. Logs are kept.
# =============================================================================
set -uo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

case "${1:-}" in
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "") ;;
    *) die "Unknown option: $1" ;;
esac

echo "🛑 CengBot System Stopping..."
echo "============================="

read -r -a SAVED <<< "$(pidfile_read)"
for i in "${!CENGBOT_SERVICES[@]}"; do
    svc="${CENGBOT_SERVICES[$i]}"
    pids=()
    unit=""
    case "$svc" in worker) unit=cengbot-worker ;; api) unit=cengbot-admin ;; bot) unit=cengbot-bot ;; esac
    if [[ -n "$unit" ]] && systemd_unit_active_here "$unit"; then
        # killing it would only make systemd restart it (Restart=always)
        print_warning "$(service_label "$svc") is managed by systemd - stop it with: sudo systemctl stop $unit"
        continue
    fi
    saved="${SAVED[$i]:--}"
    if [[ "$saved" != "-" ]] && pid_matches_service "$saved" "$svc"; then
        pids+=("$saved")
    elif [[ "$saved" != "-" ]] && pid_alive "$saved"; then
        print_warning "PID $saved from .pids is no longer $(service_label "$svc") - not touching it"
    fi
    while IFS= read -r pid; do
        [[ -n "$pid" && " ${pids[*]} " != *" $pid "* ]] && pids+=("$pid")
    done < <(service_pids "$svc")
    if (( ${#pids[@]} > 0 )); then
        timeout=15
        [[ "$svc" == "worker" ]] && timeout=60   # give the worker time to free the GPU
        stop_pids "$(service_label "$svc")" "$timeout" "${pids[@]}"
    else
        print_status "$(service_label "$svc"): not running"
    fi
done

mapfile -t legacy < <(service_pids legacy_bot)
if (( ${#legacy[@]} > 0 )); then
    stop_pids "$(service_label legacy_bot)" 15 "${legacy[@]}"
fi

rm -f -- "$PID_FILE"

remaining=0
for svc in "${CENGBOT_SERVICES[@]}" legacy_bot; do
    if [[ -n "$(service_pids "$svc")" ]]; then
        remaining=1
        print_warning "$(service_label "$svc") still running: $(service_pids "$svc" | tr '\n' ' ')"
    fi
done

echo ""
if (( remaining == 0 )); then
    print_success "All CengBot services stopped"
else
    print_warning "Some processes may still be running (ps -ef | grep -E 'ai_model_worker|admin_rest_api|telegram_bot|bin/vite( |$)')"
fi
echo "📊 Restart: ./scripts/start_system.sh      📝 Logs are preserved in logs/"
exit "$remaining"

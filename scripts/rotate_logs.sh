#!/usr/bin/env bash
# =============================================================================
# CengBot log rotation helper (no root needed)
#
# Renders config/logrotate/cengbot.conf for this project and runs logrotate as
# the current user with its own state file (logs/.logrotate.state):
#   size 50M, rotate 5, copytruncate, compress -> logs/<name>.log.1.gz ... .5.gz
#
# Usage:
#   scripts/rotate_logs.sh                  rotate now (only logs bigger than 50M)
#   scripts/rotate_logs.sh --debug          logrotate -d: show what would happen, change nothing
#   scripts/rotate_logs.sh --install-cron   add an hourly entry to YOUR crontab (asks first)
#   scripts/rotate_logs.sh --remove-cron    remove that crontab entry
#   scripts/rotate_logs.sh --print-config [--system]
#                                           print the rendered config (--system: with the
#                                           "su <user> <group>" line for /etc/logrotate.d)
# Options: -y/--yes (no questions), -q/--quiet
#
# The root-run variant (/etc/logrotate.d/cengbot + /etc/cron.hourly) is
# installed by: sudo scripts/optimize_linux.sh --only logrotate
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TEMPLATE="${CENGBOT_LOGROTATE_TEMPLATE:-$SCRIPTS_DIR/../config/logrotate/cengbot.conf}"
RENDERED="$LOG_DIR/.logrotate.conf"
STATE="$LOG_DIR/.logrotate.state"
CRON_MARK="# cengbot-logrotate"

MODE="run"; SYSTEM=0; QUIET=0
ASSUME_YES="${ASSUME_YES:-0}"
while (( $# > 0 )); do
    case "$1" in
        --debug|-d|--dry-run|-n) MODE="debug" ;;
        --install-cron)  MODE="install-cron" ;;
        --remove-cron)   MODE="remove-cron" ;;
        --print-config)  MODE="print" ;;
        --system)        SYSTEM=1 ;;
        -y|--yes)        ASSUME_YES=1 ;;
        -q|--quiet)      QUIET=1 ;;
        -h|--help)       sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               die "Unknown option: $1 (see --help)" ;;
    esac
    shift
done
export ASSUME_YES

[[ -f "$TEMPLATE" ]] || die "logrotate template not found: $TEMPLATE"

# render_config [system] -> rendered config on stdout
render_config() {
    local su_line="" user group
    if [[ "${1:-0}" == "1" ]]; then
        user="${CENGBOT_SERVICE_USER:-$(stat -c %U -- "$PROJECT_ROOT")}"
        group="$(id -gn "$user" 2>/dev/null || echo "$user")"
        su_line="su $user $group"
    fi
    local esc
    esc=$(printf '%s' "$PROJECT_ROOT" | sed -e 's/[\\|&]/\\&/g')   # escape for the sed replacement
    sed -e "s|@PROJECT_DIR@|$esc|g" -e "s|@SU_LINE@|$su_line|g" -- "$TEMPLATE"
}

logrotate_bin() {
    command -v logrotate 2>/dev/null || { [[ -x /usr/sbin/logrotate ]] && echo /usr/sbin/logrotate; } || true
}

case "$MODE" in
    print)
        render_config "$SYSTEM"
        exit 0 ;;
    install-cron|remove-cron)
        command -v crontab >/dev/null 2>&1 || die "crontab is not installed"
        current=$(crontab -l 2>/dev/null || true)
        filtered=$(printf '%s\n' "$current" | grep -vF -- "$CRON_MARK" || true)
        if [[ "$MODE" == "remove-cron" ]]; then
            printf '%s\n' "$filtered" | sed '/^$/d' | crontab -
            print_success "Removed the CengBot log rotation entry from your crontab"
            exit 0
        fi
        line="17 * * * * $(printf '%q' "$SCRIPTS_DIR/rotate_logs.sh") --quiet $CRON_MARK"
        if [[ -n "${CENGBOT_ROOT:-}" ]]; then
            line="17 * * * * CENGBOT_ROOT=$(printf '%q' "$PROJECT_ROOT") $(printf '%q' "$SCRIPTS_DIR/rotate_logs.sh") --quiet $CRON_MARK"
        fi
        echo "Crontab line: $line"
        confirm "Add this hourly entry to the crontab of user $(id -un)?" || die "Cancelled."
        { printf '%s\n' "$filtered" | sed '/^$/d'; echo "$line"; } | crontab -
        print_success "Installed (crontab -l to check, --remove-cron to undo)"
        exit 0 ;;
esac

LR=$(logrotate_bin)
if [[ -z "$LR" ]]; then
    print_warning "logrotate is not installed (sudo apt install logrotate) - falling back to the built-in size check"
    if [[ "$MODE" == "debug" ]]; then
        exit 0
    fi
    rotate_logs_dir "$LOG_DIR"
    exit 0
fi

if [[ "$MODE" == "debug" ]]; then
    # nothing is written into the project: config + state live in a temp dir
    tmpdir=$(mktemp -d)
    trap 'rm -rf -- "$tmpdir"' EXIT
    render_config 0 > "$tmpdir/cengbot.conf"
    "$LR" -d -s "$tmpdir/state" "$tmpdir/cengbot.conf"
    exit 0
fi
mkdir -p "$LOG_DIR"
# Safety net: logrotate's copytruncate copies the WHOLE file before truncating it,
# so a runaway multi-GB log would need that much free disk (and if the copy fails
# the log is not truncated). Trim such logs with tail first - same naming
# (<log>.N.gz) as logrotate's compress, so the generations stay consistent.
PRETRIM_BYTES="${CENGBOT_LOG_PRETRIM_BYTES:-1073741824}"   # 1 GiB
for f in "$LOG_DIR"/*.log; do
    [[ -e "$f" ]] || continue
    case "${f##*/}" in
        error.log|system_monitor.log) continue ;;   # rotated by Python
    esac
    rotate_log "$f" "$PRETRIM_BYTES"
done
render_config 0 > "$RENDERED.tmp"
mv -f -- "$RENDERED.tmp" "$RENDERED"
if (( QUIET == 1 )); then
    "$LR" -s "$STATE" "$RENDERED"
else
    "$LR" -v -s "$STATE" "$RENDERED"
fi

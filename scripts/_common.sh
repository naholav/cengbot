#!/usr/bin/env bash
# =============================================================================
# CengBot - shared helpers for the scripts in this directory.
#
# Source it (do not execute it):
#     # shellcheck source=scripts/_common.sh
#     source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Provides
#   PROJECT_ROOT   absolute physical path of the project (override: CENGBOT_ROOT)
#   SCRIPTS_DIR    directory containing this file
#   PYTHON         project interpreter (override: CENGBOT_PYTHON,
#                  default: $PROJECT_ROOT/.venv/bin/python)
#   print_* / die / confirm / as_root / rotate_log / rotate_logs_dir
#   service_pids / stop_pids / start_service / pidfile helpers
#
# Works under `set -euo pipefail` and without it.
# =============================================================================

if [[ -n "${_CENGBOT_COMMON_LOADED:-}" ]]; then
    return 0
fi
_CENGBOT_COMMON_LOADED=1

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -n "${CENGBOT_ROOT:-}" ]]; then
    if [[ ! -d "$CENGBOT_ROOT" ]]; then
        echo "[ERROR] CENGBOT_ROOT is set but is not a directory: $CENGBOT_ROOT" >&2
        exit 1
    fi
    PROJECT_ROOT="$(cd "$CENGBOT_ROOT" && pwd -P)"
else
    PROJECT_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd -P)"
fi

PYTHON="${CENGBOT_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.pids"

# Size-based rotation of the stdout logs the scripts redirect to logs/*.log
CENGBOT_LOG_MAX_BYTES="${CENGBOT_LOG_MAX_BYTES:-52428800}"    # rotate above 50 MiB
CENGBOT_LOG_KEEP_BYTES="${CENGBOT_LOG_KEEP_BYTES:-10485760}"  # keep last 10 MiB in .1.gz
CENGBOT_LOG_GENERATIONS="${CENGBOT_LOG_GENERATIONS:-5}"       # .1.gz .. .5.gz (as logrotate)

# ---- output -----------------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'; NC=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; BLUE=""; NC=""
fi
export RED GREEN YELLOW BLUE NC

print_status()  { printf '%s[INFO]%s %s\n' "$BLUE" "$NC" "$*"; }
print_success() { printf '%s[OK]%s %s\n' "$GREEN" "$NC" "$*"; }
print_warning() { printf '%s[WARNING]%s %s\n' "$YELLOW" "$NC" "$*"; }
print_error()   { printf '%s[ERROR]%s %s\n' "$RED" "$NC" "$*" >&2; }
print_header() {
    printf '\n%s================================================%s\n' "$BLUE" "$NC"
    printf '%s%s%s\n' "$BLUE" "$*" "$NC"
    printf '%s================================================%s\n\n' "$BLUE" "$NC"
}
die() { print_error "$*"; exit 1; }

human_size() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || printf '%s bytes' "$1"
}

# confirm "Question?"  -> 0 = yes, 1 = no.
# ASSUME_YES=1 answers yes. Without a terminal the answer is "no" (fail safe).
confirm() {
    local reply
    if [[ "${ASSUME_YES:-0}" == "1" ]]; then
        return 0
    fi
    if [[ ! -t 0 ]]; then
        print_warning "No interactive terminal - treating as 'no': $1 (pass --yes to confirm)"
        return 1
    fi
    read -r -p "$1 [y/N]: " reply || return 1
    [[ "$reply" =~ ^([yY]|[yY][eE][sS]|[eE]|[eE][vV][eE][tT])$ ]]
}

require_python() {
    if [[ ! -x "$PYTHON" ]]; then
        print_error "Python interpreter not found: $PYTHON"
        print_error "Create the project venv (.venv) or set CENGBOT_PYTHON=/path/to/python"
        exit 1
    fi
}

# Run a command as root: directly when already root, else via interactive sudo.
# Never pipes a password into sudo.
as_root() {
    if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
        "$@"
        return
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        print_error "Root privileges are required for: $*"
        print_error "'sudo' is not available - run this command as root."
        return 1
    fi
    sudo -- "$@"
}

# Re-execute the current script as root (interactive sudo), keeping arguments.
reexec_as_root() {
    if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
        return 0
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        die "This script must run as root and 'sudo' is not available. Run it as root."
    fi
    print_status "Root privileges required - re-running with sudo (you may be asked for YOUR password)..."
    exec sudo -- "$0" "$@"
}

# ---- log rotation -------------------------------------------------------------
# Regular rotation is done by logrotate (config/logrotate/cengbot.conf via
# scripts/rotate_logs.sh or /etc/logrotate.d/cengbot): size 50M, rotate 5,
# copytruncate, compress -> <log>.1.gz .. <log>.5.gz.
#
# rotate_log FILE [MAX_BYTES] [KEEP_BYTES] [GENERATIONS]  - start-time safety net
# with the SAME naming: if FILE is bigger than MAX_BYTES, shift <log>.N.gz up by
# one (dropping the oldest), store only the last KEEP_BYTES of FILE (tail -c:
# never reads a huge file completely) gzipped as <log>.1.gz and truncate FILE
# in place (safe for O_APPEND writers).
rotate_log() {
    local f="$1"
    local max="${2:-$CENGBOT_LOG_MAX_BYTES}"
    local keep="${3:-$CENGBOT_LOG_KEEP_BYTES}"
    local gens="${4:-$CENGBOT_LOG_GENERATIONS}"
    local size i tmp
    [[ -n "$f" && -f "$f" && ! -L "$f" ]] || return 0
    size=$(stat -c %s -- "$f" 2>/dev/null) || return 0
    (( size > max )) || return 0
    (( gens >= 1 )) || gens=1
    if ! command -v gzip >/dev/null 2>&1; then
        print_warning "gzip not found - not rotating $f"
        return 0
    fi
    tmp="$f.rotating.tmp"
    if ! {
        printf '[%s] --- log rotated by CengBot scripts: original size %s bytes, only the last %s bytes were kept ---\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" "$size" "$keep"
        tail -c "$keep" -- "$f"
    } | gzip -c > "$tmp" 2>/dev/null; then
        rm -f -- "$tmp"
        print_warning "Could not rotate $f (disk full?) - left untouched"
        return 0
    fi
    rm -f -- "$f.$gens.gz"
    for (( i = gens - 1; i >= 1; i-- )); do
        if [[ -f "$f.$i.gz" ]]; then
            mv -f -- "$f.$i.gz" "$f.$(( i + 1 )).gz"
        fi
    done
    mv -f -- "$tmp" "$f.1.gz"
    : > "$f"
    print_status "Rotated $(basename "$f") ($(human_size "$size")) -> $(basename "$f").1.gz (last $(human_size "$keep") kept)"
}

# rotate_logs_dir [DIR]  - rotate every DIR/*.log (default: logs/)
# Python-managed logs (RotatingFileHandler) are skipped: error.log, system_monitor.log
rotate_logs_dir() {
    local dir="${1:-$LOG_DIR}" f
    [[ -d "$dir" ]] || return 0
    for f in "$dir"/*.log; do
        [[ -e "$f" ]] || continue
        case "${f##*/}" in
            error.log|system_monitor.log) continue ;;
        esac
        rotate_log "$f"
    done
}

# node_ok -> true if Node.js satisfies admin_frontend/package.json "engines"
# (^22.13 or >=24, required by Vite/Vitest). Distribution packages (apt) are
# usually far older, so the hint points to nvm / NodeSource instead.
node_ok() {
    command -v node >/dev/null 2>&1 || return 1
    node -e 'const [a, b] = process.versions.node.split(".").map(Number); process.exit(a >= 24 || (a === 22 && b >= 13) ? 0 : 1)' 2>/dev/null
}
# shellcheck disable=SC2034  # used by the scripts that source this file
NODE_INSTALL_HINT="Install Node.js 22 LTS (>= 22.13) or 24+, e.g. with nvm: https://github.com/nvm-sh/nvm (nvm install 22), or from https://nodejs.org"

# resolve_db_file -> prints the SQLite file the services use (honours DATABASE_URL
# via src/database_models.resolve_database_url); falls back to <root>/university_bot.db.
resolve_db_file() {
    local out=""
    if [[ -x "$PYTHON" ]]; then
        out=$(cd "$PROJECT_ROOT" && "$PYTHON" - <<'PY' 2>/dev/null
import sys
sys.path.insert(0, "src")
from database_models import DATABASE_URL as url
print(url[len("sqlite:///"):] if url.startswith("sqlite:///") else "")
PY
) || out=""
    fi
    echo "${out:-$PROJECT_ROOT/university_bot.db}"
}

# ---- services / processes ---------------------------------------------------
CENGBOT_SERVICES=(worker api bot frontend)

service_label() {
    case "$1" in
        worker)   echo "AI Worker" ;;
        api)      echo "Admin API" ;;
        bot)      echo "Telegram Bot" ;;
        frontend) echo "React Frontend" ;;
        legacy_bot) echo "Legacy Telegram bot (telegram_bot.py / telegram_bot_standalone.py)" ;;
        *)        echo "$1" ;;
    esac
}

service_script() {
    case "$1" in
        worker) echo "src/ai_model_worker.py" ;;
        api)    echo "src/admin_rest_api.py" ;;
        bot)    echo "src/telegram_bot_rabbitmq.py" ;;
        *)      return 1 ;;
    esac
}

service_log() {
    case "$1" in
        worker)   echo "$LOG_DIR/worker.log" ;;
        api)      echo "$LOG_DIR/admin.log" ;;
        bot)      echo "$LOG_DIR/bot.log" ;;
        frontend) echo "$LOG_DIR/frontend.log" ;;
        *)        return 1 ;;
    esac
}

# Extended regex matched against the space-joined command line.
service_regex() {
    case "$1" in
        worker)   echo '(^|[ /])ai_model_worker\.py( |$)' ;;
        api)      echo '(^|[ /])admin_rest_api\.py( |$)' ;;
        bot)      echo '(^|[ /])telegram_bot_rabbitmq\.py( |$)' ;;
        # old polling bots - must not run next to the production bot (same token)
        legacy_bot) echo '(^|[ /])telegram_bot(_standalone)?\.py( |$)' ;;
        # the Vite dev server only (npm start -> sh -c vite -> node .../.bin/vite),
        # never "vite build", "vite preview" or vitest runs
        frontend) echo '(^|[ /])vite(\.js)?( (dev|serve))?( -|$)|(^|[ /])npm( run)? start( |$)' ;;
        *)        return 1 ;;
    esac
}

# pid_matches_service PID SERVICE -> 0 if PID is that service *of this project*
# (command line matches, python interpreter for python services, and the
# process runs inside PROJECT_ROOT or names it on its command line). This
# protects against PID reuse and against killing other users'/projects' processes.
pid_matches_service() {
    local pid="$1" svc="$2" regex cmd argv0 cwd
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    [[ "$pid" != "$$" ]] || return 1
    regex=$(service_regex "$svc") || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || return 1
    cmd="${cmd% }"
    [[ -n "$cmd" ]] || return 1
    [[ "$cmd" =~ $regex ]] || return 1
    argv0="${cmd%% *}"
    argv0="${argv0##*/}"
    if [[ "$svc" == "frontend" ]]; then
        # npm / node / the "sh -c vite" child - never a user's shell
        [[ "$argv0" =~ ^(npm|node|nodejs|sh|dash)$ ]] || return 1
    else
        [[ "$argv0" == python* ]] || return 1
    fi
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
    [[ "$cmd" == *"$PROJECT_ROOT/"* || "$cwd" == "$PROJECT_ROOT" || "$cwd" == "$PROJECT_ROOT/"* ]]
}

# service_pids SERVICE -> PIDs (one per line) of that service of this project
service_pids() {
    local svc="$1" pid regex
    regex=$(service_regex "$svc") || return 0
    command -v pgrep >/dev/null 2>&1 || return 0
    for pid in $(pgrep -f -- "$regex" 2>/dev/null || true); do
        if pid_matches_service "$pid" "$svc"; then
            echo "$pid"
        fi
    done
}

# pid_alive PID -> 0 if the process exists and is not a zombie (exited, not reaped yet)
pid_alive() {
    local state=""
    kill -0 "$1" 2>/dev/null || return 1
    if [[ -r "/proc/$1/stat" ]]; then
        state=$(sed -E 's/^.*\) ([A-Za-z]).*$/\1/' "/proc/$1/stat" 2>/dev/null || true)
    fi
    [[ "$state" != "Z" ]]
}

# stop_pids LABEL TIMEOUT_SECONDS PID... : SIGTERM, wait, then SIGKILL
stop_pids() {
    local label="$1" timeout="$2" pid waited=0
    shift 2
    local alive=("$@")
    (( ${#alive[@]} > 0 )) || return 0
    print_status "Stopping $label (PID: ${alive[*]})..."
    kill -TERM "${alive[@]}" 2>/dev/null || true
    while (( waited < timeout )); do
        local still=()
        for pid in "${alive[@]}"; do
            if pid_alive "$pid"; then
                still+=("$pid")
            fi
        done
        alive=("${still[@]+"${still[@]}"}")
        (( ${#alive[@]} > 0 )) || return 0
        sleep 1
        waited=$(( waited + 1 ))
    done
    print_warning "$label did not exit within ${timeout}s - sending SIGKILL to ${alive[*]}"
    kill -KILL "${alive[@]}" 2>/dev/null || true
}

# start_service SERVICE -> starts it in the background (stdout+stderr APPENDED to
# its log after size-based rotation); sets LAST_PID.
# fds 8/9 are closed for the child: they carry the flock()s of train_model.sh /
# switch_model.sh, and a long-lived service must never inherit (and hold) them.
# shellcheck disable=SC2034  # read by the scripts that source this file
LAST_PID=""
start_service() {
    local svc="$1" log script
    log=$(service_log "$svc") || die "unknown service: $svc"
    mkdir -p "$LOG_DIR"
    rotate_log "$log"
    if [[ "$svc" == "frontend" ]]; then
        ( cd "$PROJECT_ROOT/admin_frontend" && BROWSER=none exec nohup npm start >> "$log" 2>&1 < /dev/null 8>&- 9>&- ) &
    else
        script=$(service_script "$svc")
        ( cd "$PROJECT_ROOT" && PYTHONUNBUFFERED=1 exec nohup "$PYTHON" "$script" >> "$log" 2>&1 < /dev/null 8>&- 9>&- ) &
    fi
    # shellcheck disable=SC2034
    LAST_PID=$!
}

# .pids holds "WORKER API BOT FRONTEND" on one line (order of CENGBOT_SERVICES).
pidfile_read() {   # pidfile_read -> prints the 4 fields, "-" for missing
    local w="" a="" b="" f=""
    if [[ -f "$PID_FILE" ]]; then
        read -r w a b f _ < "$PID_FILE" || true
    fi
    echo "${w:--} ${a:--} ${b:--} ${f:--}"
}

pidfile_write() {  # pidfile_write WORKER API BOT FRONTEND
    printf '%s %s %s %s\n' "${1:--}" "${2:--}" "${3:--}" "${4:--}" > "$PID_FILE.tmp"
    mv -f -- "$PID_FILE.tmp" "$PID_FILE"
}

pidfile_set() {    # pidfile_set SERVICE PID
    local svc="$1" pid="$2" fields i
    read -r -a fields <<< "$(pidfile_read)"
    for i in "${!CENGBOT_SERVICES[@]}"; do
        if [[ "${CENGBOT_SERVICES[$i]}" == "$svc" ]]; then
            fields[i]="$pid"
        fi
    done
    pidfile_write "${fields[@]}"
}

# systemd_unit_active_here UNIT -> 0 if UNIT is active AND runs this project
systemd_unit_active_here() {
    local unit="$1" wd
    command -v systemctl >/dev/null 2>&1 || return 1
    systemctl is-active --quiet "$unit" 2>/dev/null || return 1
    wd=$(systemctl show -p WorkingDirectory --value "$unit" 2>/dev/null) || return 1
    [[ -n "$wd" && -d "$wd" ]] || return 1
    [[ "$(cd "$wd" && pwd -P)" == "$PROJECT_ROOT" ]]
}

# read_env_key KEY -> value of KEY from $PROJECT_ROOT/.env (last occurrence,
# surrounding quotes removed). Only use this for non-secret keys.
read_env_key() {
    local key="$1" line
    [[ -f "$PROJECT_ROOT/.env" ]] || return 0
    line=$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$PROJECT_ROOT/.env" 2>/dev/null | tail -n 1 || true)
    line="${line#*=}"
    line="${line%$'\r'}"
    line="${line%\"}"; line="${line#\"}"
    line="${line%\'}"; line="${line#\'}"
    printf '%s' "$line"
}

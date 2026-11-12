#!/usr/bin/env bash
#
# CengBot Data Backup Script
# ==========================
# Creates backups/backup_<timestamp>.tar.gz containing:
#   - a consistent snapshot of the SQLite database (sqlite3 ".backup" or the
#     Python online backup API: includes what is still in the -wal file)
#   - data/ (training data)
#   - models/ (adapters; the active-model symlink is kept as a symlink)
#   - logs/: small logs completely, big logs only their last part (tail -c)
#   - .env.example (never .env - it contains secrets)
# Keeps the 5 most recent archives (KEEP_BACKUPS=N to change).
#
# Usage: scripts/backup_data.sh [--no-models] [--keep N] [--dry-run]
#
# Author: naholav

set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BACKUPS_DIR="$PROJECT_ROOT/backups"
KEEP_BACKUPS="${KEEP_BACKUPS:-5}"
LOG_TAIL_BYTES="${LOG_TAIL_BYTES:-10485760}"   # big logs: keep last 10 MiB
INCLUDE_MODELS=1
DRY_RUN=0

while (( $# > 0 )); do
    case "$1" in
        --no-models) INCLUDE_MODELS=0 ;;
        --keep)      shift; KEEP_BACKUPS="${1:-}" ;;
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help)   sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           die "Unknown option: $1" ;;
    esac
    shift
done
[[ "$KEEP_BACKUPS" =~ ^[1-9][0-9]*$ ]] || die "--keep needs a positive number"

TS=$(date +%Y%m%d_%H%M%S)
NAME="backup_$TS"
STAGING="$BACKUPS_DIR/.staging_$NAME"
ARCHIVE="$BACKUPS_DIR/$NAME.tar.gz"
DB_FILE="$(resolve_db_file)"

echo -e "${GREEN}🔄 CengBot Data Backup Script${NC}"
echo -e "${GREEN}==============================${NC}"
echo "Archive: $ARCHIVE"

if (( DRY_RUN == 1 )); then
    print_status "[dry-run] would back up: university_bot.db, data/, $( (( INCLUDE_MODELS )) && echo 'models/, ')logs/ (big logs: last $(human_size "$LOG_TAIL_BYTES")), .env.example"
    print_status "[dry-run] would keep the $KEEP_BACKUPS newest archives in $BACKUPS_DIR"
    exit 0
fi

[[ -e "$ARCHIVE" ]] && die "$ARCHIVE already exists - refusing to overwrite it"
mkdir -p "$BACKUPS_DIR"
mkdir "$STAGING"                       # fails if it already exists (no merging)
cleanup_staging() {
    if [[ -n "${STAGING:-}" && "$STAGING" == "$BACKUPS_DIR"/.staging_backup_* && -d "$STAGING" ]]; then
        rm -rf -- "$STAGING"
    fi
}
trap cleanup_staging EXIT

# Database: consistent online snapshot even while the bot is writing (WAL mode:
# a plain copy of the .db file would miss the commits still in its -wal file)
echo -e "\n${YELLOW}📦 Backing up database...${NC}"
DB_NOTE=""
if [[ -f "$DB_FILE" ]]; then
    rc=0
    sqlite_snapshot "$DB_FILE" "$STAGING/university_bot.db" || rc=$?
    case "$rc" in
        0) echo -e "${GREEN}✓ Database backed up (online snapshot)${NC}" ;;
        2) DB_NOTE="university_bot.db-wal in this archive belongs to university_bot.db: restore both together."
           print_warning "Database copied without an online snapshot (.db + -wal)" ;;
        *) die "Could not back up the database $DB_FILE" ;;
    esac
else
    echo -e "${RED}✗ Database not found${NC}"
fi

echo -e "\n${YELLOW}📦 Backing up training data...${NC}"
if [[ -d "$PROJECT_ROOT/data" ]]; then
    cp -a -- "$PROJECT_ROOT/data" "$STAGING/"
    echo -e "${GREEN}✓ Training data backed up${NC}"
else
    echo -e "${RED}✗ Training data directory not found${NC}"
fi

if (( INCLUDE_MODELS == 1 )); then
    echo -e "\n${YELLOW}📦 Backing up models...${NC}"
    if [[ -d "$PROJECT_ROOT/models" ]]; then
        cp -a -- "$PROJECT_ROOT/models" "$STAGING/"     # -a keeps active-model as a symlink
        echo -e "${GREEN}✓ Models backed up${NC}"
    else
        echo -e "${RED}✗ Models directory not found${NC}"
    fi
fi

echo -e "\n${YELLOW}📦 Backing up logs...${NC}"
if [[ -d "$PROJECT_ROOT/logs" ]]; then
    mkdir -p "$STAGING/logs"
    while IFS= read -r -d '' f; do
        rel="${f#"$PROJECT_ROOT/logs/"}"
        mkdir -p "$STAGING/logs/$(dirname "$rel")"
        size=$(stat -c %s -- "$f")
        if (( size > LOG_TAIL_BYTES )); then
            tail -c "$LOG_TAIL_BYTES" -- "$f" > "$STAGING/logs/$rel.tail"
            echo "  $rel: $(human_size "$size") - only the last $(human_size "$LOG_TAIL_BYTES") saved"
        else
            cp -p -- "$f" "$STAGING/logs/$rel"
        fi
    done < <(find "$PROJECT_ROOT/logs" -type f -print0)
    echo -e "${GREEN}✓ Logs backed up${NC}"
else
    echo -e "${RED}✗ Logs directory not found${NC}"
fi

if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
    cp -p -- "$PROJECT_ROOT/.env.example" "$STAGING/"
fi

cat > "$STAGING/backup_info.txt" <<EOF
CengBot Backup Information
==========================
Backup Date: $(date)
Project Root: $PROJECT_ROOT
Active model: $(readlink "$PROJECT_ROOT/models/active-model" 2>/dev/null || echo unknown)

Contents: university_bot.db, data/, $( (( INCLUDE_MODELS )) && echo 'models/, ')logs/ (*.tail = last part of a big log), .env.example
(.env is NOT included - it contains secrets; back it up separately and securely.)

To restore:
1. Stop every service (scripts/stop_system.sh, or for systemd:
   sudo systemctl stop cengbot-worker cengbot-bot cengbot-admin cengbot-monitor).
2. Delete the WAL files next to the database: $DB_FILE-wal and $DB_FILE-shm.
   SQLite replays a leftover -wal file into the restored database and would
   overwrite or empty its tables.
3. Extract the archive and copy university_bot.db to $DB_FILE, and data/,
   models/ ... back to the project root.
4. Start the system again.
$DB_NOTE
EOF

echo -e "\n${YELLOW}🗜️ Creating compressed archive...${NC}"
tar -C "$STAGING" -czf "$ARCHIVE.partial" .
mv -- "$ARCHIVE.partial" "$ARCHIVE"
cleanup_staging
trap - EXIT
echo -e "${GREEN}✓ Archive created: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))${NC}"

echo -e "\n${YELLOW}🧹 Keeping the $KEEP_BACKUPS newest archives...${NC}"
mapfile -t old < <(find "$BACKUPS_DIR" -maxdepth 1 -type f -name 'backup_*.tar.gz' -printf '%T@ %p\n' \
                   | sort -rn | tail -n +"$(( KEEP_BACKUPS + 1 ))" | cut -d' ' -f2-)
for f in "${old[@]}"; do
    [[ -n "$f" && "$f" == "$BACKUPS_DIR"/backup_*.tar.gz ]] || continue
    rm -f -- "$f"
    echo "  removed old backup: $(basename "$f")"
done
REMAINING=$(find "$BACKUPS_DIR" -maxdepth 1 -type f -name 'backup_*.tar.gz' | wc -l)
echo -e "${GREEN}✅ Backup completed - $REMAINING archive(s) in $BACKUPS_DIR${NC}"

#!/usr/bin/env bash
#
# CengBot System Cleanup Script
#
# Default (safe): removes Python caches (__pycache__, *.pyc, *.pyo) from the
# project's own code directories only (src, config, scripts, tests) - never
# from .venv, node_modules, models, model_cache, data or backups.
#
# Optional actions (each asks for confirmation unless --yes):
#   --temp-files          delete *.tmp/*.temp/*~ in the project's own code dirs
#   --clear-logs          TRUNCATE logs/*.log (also frees space held by running services)
#   --clean-node-modules  remove admin_frontend/node_modules (npm install restores it)
#   --clean-build         remove admin_frontend/build (npm run build restores it)
#   --clean-model-cache   remove model_cache/ (base model must be downloaded again!)
#   -n, --dry-run         only show what would be removed
#   -y, --yes             do not ask
#
# Example: scripts/cleanup_system.sh --clear-logs --dry-run

set -uo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

DRY_RUN=0
DO_TEMP=0; DO_LOGS=0; DO_NODE=0; DO_BUILD=0; DO_CACHE=0
while (( $# > 0 )); do
    case "$1" in
        --temp-files)         DO_TEMP=1 ;;
        --clear-logs)         DO_LOGS=1 ;;
        --clean-node-modules) DO_NODE=1 ;;
        --clean-build)        DO_BUILD=1 ;;
        --clean-model-cache)  DO_CACHE=1 ;;
        -n|--dry-run)         DRY_RUN=1 ;;
        -y|--yes)             ASSUME_YES=1 ;;
        -h|--help)            sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                    die "Unknown option: $1 (see --help)" ;;
    esac
    shift
done
export ASSUME_YES="${ASSUME_YES:-0}"

cd "$PROJECT_ROOT" || die "Cannot cd to $PROJECT_ROOT"
[[ -f "scripts/cleanup_system.sh" || -d "src" ]] || die "$PROJECT_ROOT does not look like the CengBot project"

echo "🧹 CengBot System Cleanup ($PROJECT_ROOT)"
echo "========================"
(( DRY_RUN == 1 )) && print_status "Dry run - nothing will be removed"

CODE_DIRS=()
for d in src config scripts tests; do [[ -d "$d" ]] && CODE_DIRS+=("$d"); done

# remove_paths LABEL PATH... (honours --dry-run)
remove_paths() {
    local label="$1" p
    shift
    if (( $# == 0 )); then
        print_status "$label: nothing to remove"
        return 0
    fi
    for p in "$@"; do
        if (( DRY_RUN == 1 )); then
            echo "  would remove: $p"
        else
            rm -rf -- "$p"
        fi
    done
    (( DRY_RUN == 1 )) || print_success "$label: removed $# item(s)"
}

# 1) Python caches (always)
if (( ${#CODE_DIRS[@]} > 0 )); then
    mapfile -t caches < <(find "${CODE_DIRS[@]}" \( -type d -name '__pycache__' -prune -print \) -o \( -type f \( -name '*.pyc' -o -name '*.pyo' \) -print \) 2>/dev/null | sort -u)
    remove_paths "Python caches" "${caches[@]+"${caches[@]}"}"
fi

# 2) temporary files
if (( DO_TEMP == 1 && ${#CODE_DIRS[@]} > 0 )); then
    mapfile -t temps < <(find "${CODE_DIRS[@]}" -type f \( -name '*.tmp' -o -name '*.temp' -o -name '*~' \) 2>/dev/null)
    if (( ${#temps[@]} > 0 )); then
        printf '  %s\n' "${temps[@]}"
        if (( DRY_RUN == 1 )) || confirm "Delete these ${#temps[@]} temporary file(s)?"; then
            remove_paths "Temporary files" "${temps[@]}"
        fi
    else
        print_status "Temporary files: none found"
    fi
fi

# 3) logs - truncate instead of rm: a deleted file that a service still has open
#    keeps using disk space until the service restarts.
if (( DO_LOGS == 1 )); then
    mapfile -t logs < <(find logs -maxdepth 1 -type f \( -name '*.log' -o -name '*.log.[0-9]*' \) 2>/dev/null | sort)
    if (( ${#logs[@]} > 0 )); then
        for f in "${logs[@]}"; do echo "  $f ($(human_size "$(stat -c %s -- "$f")"))"; done
        if (( DRY_RUN == 1 )); then
            print_status "would truncate ${#logs[@]} log file(s)"
        elif confirm "Truncate these ${#logs[@]} log file(s) to 0 bytes? Their content is lost"; then
            for f in "${logs[@]}"; do : > "$f"; done
            print_success "Log files truncated"
        fi
    else
        print_warning "No log files found"
    fi
fi

# 4) node_modules / build
if (( DO_NODE == 1 )); then
    if [[ -d admin_frontend/node_modules ]]; then
        if (( DRY_RUN == 1 )) || confirm "Remove admin_frontend/node_modules ($(du -sh admin_frontend/node_modules | cut -f1))?"; then
            remove_paths "node_modules" admin_frontend/node_modules
            (( DRY_RUN == 1 )) || print_warning "Run 'npm install' in admin_frontend/ to reinstall dependencies"
        fi
    else
        print_warning "No node_modules directory found"
    fi
fi
if (( DO_BUILD == 1 )); then
    if [[ -d admin_frontend/build ]]; then
        if (( DRY_RUN == 1 )) || confirm "Remove admin_frontend/build?"; then
            remove_paths "Frontend build" admin_frontend/build
            (( DRY_RUN == 1 )) || print_warning "Run 'npm run build' in admin_frontend/ to rebuild the frontend"
        fi
    else
        print_warning "No build directory found"
    fi
fi

# 5) model cache
if (( DO_CACHE == 1 )); then
    if [[ -d model_cache ]]; then
        print_warning "model_cache/ holds the base model ($(du -sh model_cache | cut -f1)). The worker loads it OFFLINE from here;"
        print_warning "after removal it must be downloaded again (network + HUGGING_FACE_TOKEN for the gated LLaMA model)."
        if (( DRY_RUN == 1 )) || confirm "Really remove model_cache/?"; then
            remove_paths "Model cache" model_cache
        else
            print_warning "Model cache cleanup cancelled"
        fi
    else
        print_warning "No model cache directory found"
    fi
fi

echo ""
print_success "System cleanup completed"
echo "Other options: --temp-files --clear-logs --clean-node-modules --clean-build --clean-model-cache --dry-run --yes"

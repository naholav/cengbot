#!/usr/bin/env bash
# =============================================================================
# CengBot training wrapper
#
# Checks GPU / interpreter / datasets, runs src/train_model.py with the project
# venv (all unknown options are passed through to it), logs to
# <log-dir>/training/auto_training_<ts>.log (default logs/), then validates the produced model
# directory against the model contract. It never switches the production
# model - do that afterwards with
#     scripts/switch_model.sh switch <new-model>       (see docs/MODEL_SWAP.md)
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

TRAINING_SCRIPT="$PROJECT_ROOT/src/train_model.py"
MODELS_DIR="$PROJECT_ROOT/models"
TRAIN_LOG_DIR="$LOG_DIR/training"
LOCK_FILE="$PROJECT_ROOT/.training.lock"
# train_model.py defaults (used when no --dataset is given; missing files are skipped)
DEFAULT_DATASETS=("$PROJECT_ROOT/data/merged_qa_data.jsonl" "$PROJECT_ROOT/data/cengbot_qa_augmented.jsonl")

# train_model.py DEFAULT_BASE_MODEL (used when neither --base-model nor BASE_MODEL_NAME is set)
DEFAULT_BASE_MODEL="meta-llama/Llama-3.2-3B"

DATASETS=()
FORWARD=()
OUTPUT_NAME=""
BASE_MODEL=""
CACHE_DIR=""   # --cache-dir (else MODEL_CACHE_DIR / model_cache)
DRY_RUN=0
ASSUME_YES="${ASSUME_YES:-0}"

show_help() {
    cat <<EOF
CengBot training wrapper

Usage: $(basename "$0") [wrapper options] [train_model.py options]

Wrapper options:
  -n, --dry-run        Run all checks + 'train_model.py --dry-run' (data/tokenization
                       stats only: no model, no writes)
  -y, --yes            Do not ask for confirmation
  -h, --help           Show this help (training options: $PYTHON src/train_model.py --help)

Common train_model.py options (passed through):
  --dataset FILE...    Training JSONL file(s) (default: data/merged_qa_data.jsonl
                       + data/cengbot_qa_augmented.jsonl)
  --models-dir DIR     Where the model directory is created (default: models/)
  --checkpoint-dir DIR Trainer checkpoints (default: checkpoints/)
  --log-dir DIR        Training logs + history; this wrapper's log goes to DIR/training/
                       (default: logs/)
  --max-steps N        Stop after N optimizer steps (smoke test), default: --epochs
  --resume-from-checkpoint latest|DIR
                       Continue an interrupted run (same --output-name/--checkpoint-dir)
  --output-name NAME   Model directory name (default: final-best-model-v<next>)
  --base-model ID|DIR  Base model, any Hugging Face org/name or a local directory
                       (default: BASE_MODEL_NAME from env/.env, else $DEFAULT_BASE_MODEL)
  --load-in-4bit       QLoRA: needed for 7-12B base models on a 24 GB GPU
  --lora-target-modules auto|all-linear|a,b,c   (default auto)

Output: models/final-best-model-v<N>/ (adapter_config.json + adapter_model.safetensors
+ tokenizer + training_info.json). Activate it with
  scripts/switch_model.sh switch final-best-model-v<N>
EOF
}

# abspath P -> P made absolute against the CALLER's directory (train_model.py runs in the project root)
abspath() {
    if [[ "$1" == /* ]]; then printf '%s' "$1"; else printf '%s' "$PWD/$1"; fi
}

while (( $# > 0 )); do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        -y|--yes)     ASSUME_YES=1 ;;
        -h|--help)    show_help; exit 0 ;;
        --dataset)
            shift
            while (( $# > 0 )) && [[ "$1" != -* ]]; do DATASETS+=("$(abspath "$1")"); shift; done
            (( ${#DATASETS[@]} > 0 )) || die "--dataset needs at least one file"
            continue ;;
        --dataset=*)  DATASETS+=("$(abspath "${1#--dataset=}")") ;;
        --models-dir)
            shift; [[ $# -gt 0 ]] || die "--models-dir needs a directory"
            MODELS_DIR="$(abspath "$1")"; FORWARD+=(--models-dir "$MODELS_DIR") ;;
        --models-dir=*) MODELS_DIR="$(abspath "${1#--models-dir=}")"; FORWARD+=(--models-dir "$MODELS_DIR") ;;
        --output-name)
            shift; [[ $# -gt 0 ]] || die "--output-name needs a name"
            OUTPUT_NAME="$1"; FORWARD+=(--output-name "$1") ;;
        --output-name=*) OUTPUT_NAME="${1#--output-name=}"; FORWARD+=("$1") ;;
        --base-model)
            shift; [[ $# -gt 0 ]] || die "--base-model needs a model id or directory"
            BASE_MODEL="$1"; FORWARD+=(--base-model "$1") ;;
        --base-model=*) BASE_MODEL="${1#--base-model=}"; FORWARD+=("$1") ;;
        --cache-dir)
            shift; [[ $# -gt 0 ]] || die "--cache-dir needs a directory"
            CACHE_DIR="$1"; FORWARD+=(--cache-dir "$1") ;;
        --cache-dir=*) CACHE_DIR="${1#--cache-dir=}"; FORWARD+=("$1") ;;
        --checkpoint-dir)
            shift; [[ $# -gt 0 ]] || die "--checkpoint-dir needs a directory"
            FORWARD+=(--checkpoint-dir "$(abspath "$1")") ;;
        --checkpoint-dir=*) FORWARD+=(--checkpoint-dir "$(abspath "${1#--checkpoint-dir=}")") ;;
        --log-dir)
            shift; [[ $# -gt 0 ]] || die "--log-dir needs a directory"
            TRAIN_LOG_DIR="$(abspath "$1")/training"; FORWARD+=(--log-dir "$(abspath "$1")") ;;
        --log-dir=*) TRAIN_LOG_DIR="$(abspath "${1#--log-dir=}")/training"; FORWARD+=(--log-dir "$(abspath "${1#--log-dir=}")") ;;
        *)            FORWARD+=("$1") ;;
    esac
    shift
done
export ASSUME_YES

check_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        die "nvidia-smi not found - GPU training is not available on this machine"
    fi
    print_status "GPU:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || die "nvidia-smi failed (driver problem?)"
}

# check_dataset FILE - every non-empty line must be JSON (streamed, constant memory)
check_dataset() {
    local path="$1"
    [[ -f "$path" ]] || die "Dataset not found: $path"
    "$PYTHON" -I - "$path" <<'PY' || die "Dataset check failed: $path"
import json, sys
path = sys.argv[1]
n = 0
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"{path}: invalid JSON at line {i}: {e}")
            sys.exit(1)
        n += 1
print(f"{path}: {n} JSON records")
PY
}

# expected_model_name -> the directory train_model.py will create
# (same rule as train_model.get_next_model_version: highest leading integer + 1)
expected_model_name() {
    local highest=0 d v
    if [[ -n "$OUTPUT_NAME" ]]; then
        echo "$OUTPUT_NAME"
        return
    fi
    for d in "$MODELS_DIR"/final-best-model-v*; do
        [[ -d "$d" ]] || continue
        v="${d##*/final-best-model-v}"
        [[ "$v" =~ ^([0-9]+)(\.[0-9]+)*$ ]] || continue
        v="${BASH_REMATCH[1]}"
        if (( 10#$v > highest )); then
            highest=$(( 10#$v ))
        fi
    done
    echo "final-best-model-v$(( highest + 1 ))"
}

# check_base_model_cached - warn when the base model (any org/name) is not in the HF cache
check_base_model_cached() {
    local base="${BASE_MODEL:-${BASE_MODEL_NAME:-$(read_env_key BASE_MODEL_NAME)}}" cache token allow
    base="${base:-$DEFAULT_BASE_MODEL}"
    if [[ "$base" == /* || "$base" == ./* || "$base" == ../* || "$base" == "~"* || -d "$base" ]]; then
        return 0    # local directory: nothing to download
    fi
    cache="${CACHE_DIR:-${MODEL_CACHE_DIR:-$(read_env_key MODEL_CACHE_DIR)}}"
    cache="${cache:-model_cache}"
    # same rule as switch_model.sh model_cache_dir / train_model.py _default_cache_dir
    # shellcheck disable=SC2088  # a literal "~/" prefix is expanded by hand
    if [[ "${cache:0:2}" == '~/' ]]; then
        cache="$HOME/${cache:2}"
    elif [[ "$cache" != /* ]]; then
        cache="$PROJECT_ROOT/$cache"
    fi
    if compgen -G "$cache/models--${base//\//--}/snapshots/*/config.json" >/dev/null; then
        return 0
    fi
    token="${HUGGING_FACE_TOKEN:-$(read_env_key HUGGING_FACE_TOKEN)}"
    allow="${MODEL_ALLOW_DOWNLOAD:-$(read_env_key MODEL_ALLOW_DOWNLOAD)}"
    if [[ "${allow,,}" =~ ^(0|false|no|off|disabled)$ ]]; then
        print_warning "Base model $base is not cached in $cache and MODEL_ALLOW_DOWNLOAD=$allow - training will fail until it is in the cache (allow the download once, see docs/MODEL_SWAP.md)"
        return 0
    fi
    print_warning "Base model $base is not cached in $cache - the first training run downloads it once (needs internet access to huggingface.co)"
    if [[ -z "$token" ]]; then
        print_warning "HUGGING_FACE_TOKEN is empty: downloading a gated base model (e.g. meta-llama/*, google/gemma-*) will fail"
    fi
}

main() {
    print_header "CengBot Training"
    require_python
    [[ -f "$TRAINING_SCRIPT" ]] || die "Training script not found: $TRAINING_SCRIPT"

    local train_args=() p found=0
    if (( ${#DATASETS[@]} > 0 )); then
        for p in "${DATASETS[@]}"; do check_dataset "$p"; done
        train_args+=(--dataset "${DATASETS[@]}")
    else
        for p in "${DEFAULT_DATASETS[@]}"; do
            if [[ -f "$p" ]]; then check_dataset "$p"; found=1; fi
        done
        (( found == 1 )) || die "No training data found (looked for: ${DEFAULT_DATASETS[*]})"
    fi
    train_args+=("${FORWARD[@]+"${FORWARD[@]}"}")

    check_base_model_cached

    if (( DRY_RUN == 1 )); then
        print_status "Dry run: $PYTHON $TRAINING_SCRIPT --dry-run ${train_args[*]}"
        ( cd "$PROJECT_ROOT" && "$PYTHON" "$TRAINING_SCRIPT" --dry-run "${train_args[@]}" ) || die "train_model.py --dry-run failed"
        print_success "Dry run finished - nothing was trained or written."
        return 0
    fi

    check_gpu
    if [[ -n "$(service_pids worker)" ]] || systemd_unit_active_here cengbot-worker; then
        print_warning "The AI worker is running and shares the GPU with training (possible out-of-memory)."
    fi

    local expected
    expected=$(expected_model_name)
    [[ -e "$MODELS_DIR/$expected" ]] && print_warning "$MODELS_DIR/$expected already exists - train_model.py refuses to overwrite it unless you pass --overwrite"
    print_status "Expected output: $MODELS_DIR/$expected/ (existing models are not modified)"
    print_status "Command: $PYTHON $TRAINING_SCRIPT ${train_args[*]}"
    confirm "Start training now? (takes several hours)" || die "Cancelled."

    mkdir -p "$TRAIN_LOG_DIR"
    exec 8> "$LOCK_FILE"
    if command -v flock >/dev/null 2>&1; then
        flock -n 8 || die "Another training run is in progress (lock: $LOCK_FILE)"
    fi

    local marker log_file rc statuses=()
    marker="$TRAIN_LOG_DIR/.started_$$"
    : > "$marker"
    log_file="$TRAIN_LOG_DIR/auto_training_$(date +%Y%m%d_%H%M%S).log"
    print_header "Training (log: $log_file)"
    set +e
    ( cd "$PROJECT_ROOT" && "$PYTHON" "$TRAINING_SCRIPT" "${train_args[@]}" ) 2>&1 | tee -a "$log_file"
    statuses=("${PIPESTATUS[@]}")
    set -e
    rc=${statuses[0]}
    if (( rc != 0 )); then
        rm -f -- "$marker"
        die "Training FAILED (exit code $rc) - see $log_file"
    fi
    print_success "Training process finished"

    # The model directory created by this run: train_model.py prints FINAL_MODEL_DIR=<dir>
    local new_dir=""
    new_dir=$(grep -a -o 'FINAL_MODEL_DIR=.*' "$log_file" | tail -n 1 | cut -d= -f2- | tr -d '\r' || true)
    if [[ -n "$new_dir" && -d "$new_dir" ]]; then
        :
    elif [[ -d "$MODELS_DIR/$expected" && "$MODELS_DIR/$expected" -nt "$marker" ]]; then
        new_dir="$MODELS_DIR/$expected"
    else
        new_dir=$(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -newer "$marker" -printf '%T@ %p\n' \
                  | sort -n | tail -n 1 | cut -d' ' -f2-)
    fi
    rm -f -- "$marker"
    [[ -n "$new_dir" ]] || die "Training finished but no new model directory was found in $MODELS_DIR - check $log_file"
    print_status "New model directory: $new_dir"
    if "$SCRIPTS_DIR/switch_model.sh" validate "$new_dir"; then
        print_success "The new model satisfies the model contract."
        echo
        print_status "Next steps (see docs/MODEL_SWAP.md):"
        printf '    %s/switch_model.sh switch %s\n' "$SCRIPTS_DIR" "$new_dir"
        printf '    %s/switch_model.sh rollback      # if the new model misbehaves\n' "$SCRIPTS_DIR"
    else
        die "The new model directory does not satisfy the model contract - do NOT switch to it"
    fi
}

main

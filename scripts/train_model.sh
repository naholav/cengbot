#!/usr/bin/env bash
# =============================================================================
# CengBot training wrapper
#
# Checks GPU / interpreter / datasets, runs src/train_model.py with the project
# venv (all unknown options are passed through to it), logs to
# logs/training/auto_training_<ts>.log, then validates the produced model
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

DATASETS=()
FORWARD=()
OUTPUT_NAME=""
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
  --output-name NAME   Model directory name (default: final-best-model-v<next>)

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

    if [[ -z "$(read_env_key HUGGING_FACE_TOKEN)" ]] && \
       ! compgen -G "$PROJECT_ROOT/model_cache/models--meta-llama--Llama-3.2-3B/snapshots/*/config.json" >/dev/null; then
        print_warning "Base model is not cached in model_cache/ and HUGGING_FACE_TOKEN is empty - downloading the gated base model will fail"
    fi

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

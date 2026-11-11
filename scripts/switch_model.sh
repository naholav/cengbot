#!/usr/bin/env bash
# =============================================================================
# CengBot model switcher
#
# The AI worker always loads the model behind the symlink models/active-model
# (single source of truth). This script lists, validates, switches and rolls
# back that symlink and restarts ONLY the AI worker so it loads the new model.
# Models are never deleted.
#
# Runbook: docs/MODEL_SWAP.md          Help: scripts/switch_model.sh help
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODELS_DIR="$PROJECT_ROOT/models"
ACTIVE_LINK="$MODELS_DIR/active-model"
PREVIOUS_FILE="$MODELS_DIR/.previous-model"
HISTORY_FILE="$MODELS_DIR/.switch-history.log"
LOCK_FILE="$MODELS_DIR/.switch.lock"
SELF="$SCRIPTS_DIR/switch_model.sh"

DRY_RUN=0
NO_RESTART="${CENGBOT_NO_RESTART:-0}"
ASSUME_YES="${ASSUME_YES:-0}"
MOVE=0
IMPORT_NAME=""

show_help() {
    cat <<EOF
CengBot model switcher - manages models/active-model (see docs/MODEL_SWAP.md)

Usage: $(basename "$0") [options] <command> [model]

Commands:
  list                    List models in models/ (type, size, metrics) and mark the active one
  current                 Show the active model and validate it
  validate <name|path>    Check a model directory against the model contract
  switch <name|path>      Validate, (import if outside models/), atomically repoint
                          models/active-model, record the previous model, restart the AI worker
  rollback                Switch back to the previously active model
  restart                 Restart only the AI worker (e.g. after a manual change)
  help                    Show this help

<name> is a directory in models/ (e.g. final-best-model-v2), or a version
shortcut (v2, 1.2 -> final-best-model-v2 / final-best-model-v1.2).
<path> may also point outside models/: it is validated and then copied into
models/<name> (asks for confirmation).

Options:
  -n, --dry-run     Show what would happen; change nothing
  -y, --yes         Do not ask for confirmation (import, worker restart)
      --no-restart  Do not restart the AI worker; print the command instead
                    (same as CENGBOT_NO_RESTART=1)
      --name NAME   Directory name under models/ when importing an outside path
      --move        Move (instead of copy) an outside model into models/ when it
                    is on the same filesystem (otherwise it is copied)

A valid model directory is either
  (1) a PEFT LoRA adapter: adapter_config.json + adapter_model.safetensors,
      directly or in a method1/ subdirectory (legacy v1 layout); or
  (2) a full HF model: config.json + *.safetensors (or pytorch_model*.bin) +
      tokenizer.json / tokenizer.model.

Environment: CENGBOT_ROOT (project root, default: parent of scripts/),
             CENGBOT_PYTHON (interpreter used for validation), CENGBOT_NO_RESTART=1
EOF
}

# ---------------------------------------------------------------------------
# Validation (contract g). One embedded Python program; prints KEY=VALUE lines.
# ---------------------------------------------------------------------------
read -r -d '' VALIDATOR_PY <<'PYEOF' || true
import glob, json, os, struct, sys

d, model_cache, default_base = sys.argv[1], sys.argv[2], sys.argv[3]
problems, warnings = [], []
out = {"KIND": "", "WEIGHTS": "", "BASE": "", "INFO": ""}


def rel(p):
    try:
        r = os.path.relpath(p, d)
        return os.path.basename(d) if r == "." else r
    except ValueError:
        return p


def load_json(path, required=True):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        if required:
            problems.append(f"{rel(path)}: not valid JSON ({e})")
        return None
    if not isinstance(data, dict):
        if required:
            problems.append(f"{rel(path)}: expected a JSON object")
        return None
    return data


def check_safetensors(path):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise ValueError(f"file too small ({size} bytes)")
            (n,) = struct.unpack("<Q", raw)
            if n == 0 or 8 + n > size:
                raise ValueError(f"header length {n} does not fit file size {size}")
            header = json.loads(f.read(n).decode("utf-8"))
        ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
        if not ends:
            raise ValueError("contains no tensors")
        need = 8 + n + max(ends)
        if need > size:
            raise ValueError(f"truncated: needs {need} bytes, file has {size}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"{rel(path)}: invalid safetensors file ({e})")
        return False
    return True


def lora_complete(p):
    return (os.path.isfile(os.path.join(p, "adapter_config.json"))
            and os.path.isfile(os.path.join(p, "adapter_model.safetensors")))


def has_tokenizer(p):
    return any(os.path.isfile(os.path.join(p, t)) for t in ("tokenizer.json", "tokenizer.model"))


m1 = os.path.join(d, "method1")
if not os.path.isdir(d):
    problems.append(f"not a directory: {d}")
elif lora_complete(d) or lora_complete(m1):
    direct = lora_complete(d)
    wdir = d if direct else m1
    out["KIND"] = "lora" if direct else "lora-method1"
    out["WEIGHTS"] = wdir
    if direct and lora_complete(m1):
        warnings.append("both ./ and ./method1/ contain a LoRA adapter - make sure they are the same model")
    cfg = load_json(os.path.join(wdir, "adapter_config.json"))
    check_safetensors(os.path.join(wdir, "adapter_model.safetensors"))
    base = ""
    if cfg is not None:
        if cfg.get("peft_type") not in (None, "LORA"):
            warnings.append(f"adapter peft_type is {cfg.get('peft_type')!r}, expected 'LORA'")
        base = cfg.get("base_model_name_or_path") or ""
        if not base:
            base = default_base
            warnings.append(f"adapter_config.json has no base_model_name_or_path - BASE_MODEL_NAME ({base}) will be used")
    out["BASE"] = base
    if base and not os.path.isdir(base):
        cache = os.path.join(model_cache, "models--" + base.replace("/", "--"))
        if not glob.glob(os.path.join(cache, "snapshots", "*", "config.json")):
            warnings.append(f"base model {base} is not cached in model_cache/ - loading it needs network access to huggingface.co")
    if not (has_tokenizer(wdir) or has_tokenizer(d)):
        warnings.append("no tokenizer files in the model dir - the base model tokenizer will be used")
elif os.path.isfile(os.path.join(d, "config.json")):
    out["KIND"] = "full"
    out["WEIGHTS"] = d
    load_json(os.path.join(d, "config.json"))
    st = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    bins = sorted(glob.glob(os.path.join(d, "pytorch_model*.bin")))
    if not st and not bins:
        problems.append("full model: no *.safetensors or pytorch_model*.bin weight files")
    for p in st:
        check_safetensors(p)
    for p in bins:
        if os.path.getsize(p) == 0:
            problems.append(f"{rel(p)}: empty file")
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx = os.path.join(d, index_name)
        if os.path.isfile(idx):
            data = load_json(idx) or {}
            shards = sorted(set((data.get("weight_map") or {}).values()))
            missing = [s for s in shards if not os.path.isfile(os.path.join(d, s))]
            if missing:
                problems.append(f"{index_name}: missing shard files: {', '.join(missing)}")
    if not has_tokenizer(d):
        problems.append("full model: no tokenizer.json or tokenizer.model")
else:
    found_partial = False
    for sub in (d, m1):
        cfg_ok = os.path.isfile(os.path.join(sub, "adapter_config.json"))
        st_ok = os.path.isfile(os.path.join(sub, "adapter_model.safetensors"))
        if cfg_ok and not st_ok:
            found_partial = True
            if os.path.isfile(os.path.join(sub, "adapter_model.bin")):
                problems.append(f"{rel(sub)}: only adapter_model.bin found - the contract needs adapter_model.safetensors "
                                "(re-save with save_pretrained(..., safe_serialization=True))")
            else:
                problems.append(f"{rel(sub)}: adapter_config.json present but adapter_model.safetensors is missing")
        elif st_ok and not cfg_ok:
            found_partial = True
            problems.append(f"{rel(sub)}: adapter_model.safetensors present but adapter_config.json is missing")
    if not found_partial:
        problems.append("not a model directory: expected adapter_config.json + adapter_model.safetensors "
                        "(directly or in method1/) or config.json + weights + tokenizer files")

info = load_json(os.path.join(d, "training_info.json"), required=False) if os.path.isdir(d) else None
if info:
    parts = []
    loss = info.get("final_loss")
    if isinstance(loss, (int, float)):
        parts.append(f"loss {loss:.4f}")
    elif loss is not None:
        parts.append(f"loss {loss}")
    if info.get("total_steps") is not None:
        parts.append(f"steps {info.get('total_steps')}")
    if info.get("timestamp"):
        parts.append(str(info.get("timestamp"))[:10])
    out["INFO"] = " | ".join(parts)

for k, v in out.items():
    print(f"{k}={v}")
for w in warnings:
    print(f"WARN={w}")
for p in problems:
    print(f"ERROR={p}")
sys.exit(1 if problems else 0)
PYEOF

validator_python() {
    if [[ -x "$PYTHON" ]]; then
        echo "$PYTHON"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        die "No Python interpreter found (looked for $PYTHON and python3) - needed to validate models"
    fi
}

# validate_dir DIR -> sets MODEL_KIND MODEL_BASE MODEL_WEIGHTS MODEL_INFO
#                     MODEL_WARNINGS[] MODEL_ERRORS[]; returns 0 if valid
# shellcheck disable=SC2034  # MODEL_WEIGHTS is informational (set for callers)
validate_dir() {
    local dir="$1" py output line base_default rc=0
    MODEL_KIND=""; MODEL_BASE=""; MODEL_WEIGHTS=""; MODEL_INFO=""
    MODEL_WARNINGS=(); MODEL_ERRORS=()
    py=$(validator_python)
    base_default="${BASE_MODEL_NAME:-$(read_env_key BASE_MODEL_NAME)}"
    base_default="${base_default:-meta-llama/Llama-3.2-3B}"
    output=$("$py" -I -c "$VALIDATOR_PY" "$dir" "$PROJECT_ROOT/model_cache" "$base_default" 2>&1) || rc=$?
    while IFS= read -r line; do
        case "$line" in
            KIND=*)    MODEL_KIND="${line#KIND=}" ;;
            BASE=*)    MODEL_BASE="${line#BASE=}" ;;
            WEIGHTS=*) MODEL_WEIGHTS="${line#WEIGHTS=}" ;;
            INFO=*)    MODEL_INFO="${line#INFO=}" ;;
            WARN=*)    MODEL_WARNINGS+=("${line#WARN=}") ;;
            ERROR=*)   MODEL_ERRORS+=("${line#ERROR=}") ;;
            "")        ;;
            *)         MODEL_ERRORS+=("validator: $line") ;;
        esac
    done <<< "$output"
    if (( rc != 0 )) && (( ${#MODEL_ERRORS[@]} == 0 )); then
        MODEL_ERRORS+=("validator failed with exit code $rc")
    fi
    (( ${#MODEL_ERRORS[@]} == 0 ))
}

kind_label() {
    case "$1" in
        lora)         echo "LoRA adapter" ;;
        lora-method1) echo "LoRA adapter (method1/)" ;;
        full)         echo "full model" ;;
        *)            echo "INVALID" ;;
    esac
}

# report_validation NAME  (after validate_dir)
report_validation() {
    local name="$1" w e
    if (( ${#MODEL_ERRORS[@]} == 0 )); then
        print_success "$name: valid - $(kind_label "$MODEL_KIND")${MODEL_BASE:+, base model: $MODEL_BASE}"
    else
        print_error "$name: NOT a valid model directory"
    fi
    if (( ${#MODEL_ERRORS[@]} > 0 )); then
        for e in "${MODEL_ERRORS[@]}"; do printf '    - %s\n' "$e" >&2; done
    fi
    if (( ${#MODEL_WARNINGS[@]} > 0 )); then
        for w in "${MODEL_WARNINGS[@]}"; do print_warning "$w"; done
    fi
    if [[ -n "$MODEL_INFO" ]]; then
        print_status "training info: $MODEL_INFO"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Symlink state
# ---------------------------------------------------------------------------
models_abs() {
    [[ -d "$MODELS_DIR" ]] || die "Models directory not found: $MODELS_DIR"
    (cd "$MODELS_DIR" && pwd -P)
}

# current_rel -> active-model target relative to models/ ("" if no link)
current_rel() {
    local target mabs
    [[ -L "$ACTIVE_LINK" ]] || return 0
    target=$(readlink -- "$ACTIVE_LINK")
    target="${target%/}"
    if [[ "$target" == /* ]]; then
        mabs=$(models_abs)
        if [[ "$target" == "$mabs/"* ]]; then
            target="${target#"$mabs"/}"
        elif [[ "$target" == "$MODELS_DIR/"* ]]; then
            target="${target#"$MODELS_DIR"/}"
        fi
    fi
    target="${target#./}"
    printf '%s' "$target"
}

previous_rel() {
    [[ -f "$PREVIOUS_FILE" ]] || return 0
    head -n 1 -- "$PREVIOUS_FILE" | tr -d '\r'
}

check_link_replaceable() {
    if [[ -e "$ACTIVE_LINK" && ! -L "$ACTIVE_LINK" ]]; then
        die "$ACTIVE_LINK exists and is NOT a symlink - refusing to replace it. Move it away manually first."
    fi
}

# resolve_target ARG [check|nocheck] -> sets TARGET_REL (relative to models/), TARGET_DIR,
#                       NEEDS_IMPORT (0/1) and SOURCE_DIR (when importing)
resolve_target() {
    local arg="$1" cand="" abs mabs
    [[ -n "$arg" ]] || die "Missing model name or path. Run '$(basename "$0") list'."
    mabs=$(models_abs)
    TARGET_REL=""; TARGET_DIR=""; NEEDS_IMPORT=0; SOURCE_DIR=""
    if [[ "$arg" != */* && "$arg" != "." && "$arg" != ".." ]]; then
        if [[ "$arg" == "active-model" ]]; then
            die "'active-model' is the symlink itself - give the model directory name (see '$(basename "$0") list')"
        fi
        if [[ -d "$MODELS_DIR/$arg" ]]; then
            cand="$MODELS_DIR/$arg"
        elif [[ "$arg" =~ ^[vV]?[0-9][0-9.]*$ && -d "$MODELS_DIR/final-best-model-v${arg#[vV]}" ]]; then
            cand="$MODELS_DIR/final-best-model-v${arg#[vV]}"
        fi
    fi
    if [[ -z "$cand" ]]; then
        if [[ -d "$arg" ]]; then
            cand="$arg"
        else
            die "Model not found: '$arg' (not a directory in $MODELS_DIR and not an existing path). Run '$(basename "$0") list'."
        fi
    fi
    abs=$(cd "$cand" && pwd -P)
    if [[ "$abs" == "$mabs" ]]; then
        die "'$arg' is the models directory itself - give a model directory inside it"
    fi
    if [[ "$abs" == "$mabs/"* ]]; then
        TARGET_REL="${abs#"$mabs"/}"
        if [[ "$TARGET_REL" == */method1 ]]; then
            TARGET_REL="${TARGET_REL%/method1}"
        fi
        TARGET_DIR="$mabs/$TARGET_REL"
    else
        NEEDS_IMPORT=1
        SOURCE_DIR="$abs"
        local name="$IMPORT_NAME"
        if [[ -z "$name" ]]; then
            name="$(basename "$abs")"
            if [[ "$name" == "method1" ]]; then
                name="$(basename "$(dirname "$abs")")"
            fi
        fi
        if [[ ! "$name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || "$name" == "active-model" ]]; then
            die "Invalid model name '$name' (allowed: letters, digits, . _ -; not 'active-model'). Use --name NAME."
        fi
        TARGET_REL="$name"
        TARGET_DIR="$mabs/$name"
        if [[ "${2:-check}" == "check" ]] && [[ -e "$TARGET_DIR" || -L "$TARGET_DIR" ]]; then
            die "models/$name already exists - refusing to overwrite it. Use --name NEW_NAME, or 'switch $name' if it is the same model."
        fi
    fi
}

# ---------------------------------------------------------------------------
# Import a model from outside models/ (copy, or move on the same filesystem)
# ---------------------------------------------------------------------------
IMPORT_TMP=""
cleanup_import_tmp() {
    if [[ -n "$IMPORT_TMP" && "$IMPORT_TMP" == "$MODELS_DIR"/.importing-* && -d "$IMPORT_TMP" ]]; then
        rm -rf -- "$IMPORT_TMP"
    fi
}

import_model() {
    local src="$1" name="$2" dest="$MODELS_DIR/$2" need avail
    if (( MOVE == 1 )) && [[ "$(stat -c %d -- "$src")" == "$(stat -c %d -- "$MODELS_DIR")" ]]; then
        print_status "Moving $src -> models/$name"
        mv -T -- "$src" "$dest"
        return 0
    fi
    (( MOVE == 1 )) && print_warning "--move: source is on another filesystem - copying instead (the source is left untouched)"
    need=$(du -sbL -- "$src" | cut -f1)
    avail=$(df -B1 --output=avail -- "$MODELS_DIR" | tail -n 1 | tr -d ' ')
    if [[ "$need" =~ ^[0-9]+$ && "$avail" =~ ^[0-9]+$ ]] && (( need + 104857600 > avail )); then
        die "Not enough disk space in $MODELS_DIR: need $(human_size "$need") (+100 MiB margin), available $(human_size "$avail")"
    fi
    print_status "Copying $src -> models/$name ($(human_size "$need"))..."
    # Our own partial copy is removed if the copy fails or is interrupted.
    IMPORT_TMP="$MODELS_DIR/.importing-$name.$$"
    trap cleanup_import_tmp EXIT
    cp -R -L --preserve=mode,timestamps -T -- "$src" "$IMPORT_TMP"
    mv -T -- "$IMPORT_TMP" "$dest"
    IMPORT_TMP=""
    trap - EXIT
}

# ---------------------------------------------------------------------------
# Point models/active-model at TARGET_REL atomically and record the previous one
# ---------------------------------------------------------------------------
set_active() {
    local new="$1" old="$2" tmp="$MODELS_DIR/.active-model.tmp.$$"
    rm -f -- "$tmp"
    ln -sfn -- "$new" "$tmp"
    mv -Tf -- "$tmp" "$ACTIVE_LINK"        # rename(2): atomic replacement
    if [[ -n "$old" && "$old" != "$new" ]]; then
        printf '%s\n' "$old" > "$PREVIOUS_FILE.tmp"
        mv -f -- "$PREVIOUS_FILE.tmp" "$PREVIOUS_FILE"
    fi
    printf '%s\t%s -> %s\tby %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${old:-<none>}" "$new" "${USER:-$(id -un)}" >> "$HISTORY_FILE"
}

worker_start_hint() {
    printf '    %s restart --yes\n' "$SELF"
    printf '  or (full system):  %s/start_system.sh\n' "$SCRIPTS_DIR"
}

# Restart ONLY the AI worker so it loads the new active model.
restart_worker() {
    local pids=()
    if (( DRY_RUN == 1 )); then
        if [[ "$NO_RESTART" == "1" ]]; then
            print_status "[dry-run] would NOT restart the AI worker (--no-restart)"
        else
            print_status "[dry-run] would restart the AI worker (only the worker - bot/API keep running)"
        fi
        return 0
    fi
    if [[ "$NO_RESTART" == "1" ]]; then
        print_warning "AI worker NOT restarted (--no-restart). It keeps serving the old model until restarted:"
        worker_start_hint
        return 0
    fi
    if systemd_unit_active_here cengbot-worker; then
        print_status "AI worker runs under systemd - restarting cengbot-worker.service (sudo may ask for your password)"
        if as_root systemctl restart cengbot-worker; then
            print_success "cengbot-worker restarted. Watch it load: tail -f $(service_log worker)"
            return 0
        fi
        print_error "Restart failed. Run manually: sudo systemctl restart cengbot-worker"
        return 1
    fi
    mapfile -t pids < <(service_pids worker)
    if (( ${#pids[@]} == 0 )); then
        print_warning "The AI worker is not running - the new model will be loaded when it starts:"
        worker_start_hint
        return 0
    fi
    if ! confirm "Restart the AI worker now (PID ${pids[*]})? Incoming questions wait in RabbitMQ meanwhile"; then
        print_warning "AI worker NOT restarted - it keeps serving the old model. Restart it later with:"
        worker_start_hint
        return 0
    fi
    stop_pids "AI worker" 60 "${pids[@]}"
    require_python
    start_service worker
    pidfile_set worker "$LAST_PID"
    sleep 3
    if pid_alive "$LAST_PID"; then
        print_success "AI worker restarted (PID $LAST_PID). Loading the model takes a few minutes:"
        printf '    tail -f %s\n' "$(service_log worker)"
    else
        print_error "The AI worker exited right after starting - check $(service_log worker)"
        print_error "To go back to the previous model: $SELF rollback --yes"
        return 1
    fi
}

with_lock() {
    if (( DRY_RUN == 0 )) && command -v flock >/dev/null 2>&1; then
        exec 9> "$LOCK_FILE"
        flock -n 9 || die "Another switch_model.sh is running (lock: $LOCK_FILE)"
    fi
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
cmd_list() {
    local current prev names name kind size marker shown=0
    [[ -d "$MODELS_DIR" ]] || die "Models directory not found: $MODELS_DIR"
    current=$(current_rel)
    prev=$(previous_rel)
    print_header "Models in $MODELS_DIR"
    mapfile -t names < <(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -printf '%f\n' | sort -V)
    printf '  %-2s %-30s %-24s %-9s %s\n' "" "NAME" "TYPE" "SIZE" "INFO"
    for name in "${names[@]}"; do
        [[ -n "$name" ]] || continue
        shown=1
        if validate_dir "$MODELS_DIR/$name"; then
            kind=$(kind_label "$MODEL_KIND")
        else
            kind="INVALID"
        fi
        size=$(du -sh -- "$MODELS_DIR/$name" 2>/dev/null | cut -f1)
        marker=""
        if [[ -n "$current" && "${current%%/*}" == "$name" ]]; then
            marker="*"
        fi
        local note="$MODEL_INFO"
        [[ "$name" == "$prev" ]] && note="${note:+$note  }(previous)"
        if [[ "$marker" == "*" ]]; then
            printf '%s  %-2s %-30s %-24s %-9s %s%s\n' "$GREEN" "$marker" "$name" "$kind" "$size" "${note:+$note  }(ACTIVE)" "$NC"
        else
            printf '  %-2s %-30s %-24s %-9s %s\n' "$marker" "$name" "$kind" "$size" "$note"
        fi
    done
    (( shown == 1 )) || print_warning "No model directories found in $MODELS_DIR"
    echo
    if [[ -L "$ACTIVE_LINK" ]]; then
        if [[ -e "$ACTIVE_LINK" ]]; then
            print_status "active-model -> $current"
        else
            print_error "active-model -> $current (BROKEN LINK: target missing)"
        fi
    else
        print_warning "No active model set (models/active-model missing)"
    fi
}

cmd_current() {
    local current target model_path
    current=$(current_rel)
    print_header "Active model"
    if [[ ! -L "$ACTIVE_LINK" ]]; then
        if [[ -e "$ACTIVE_LINK" ]]; then
            print_warning "$ACTIVE_LINK is a real directory, not a symlink"
        else
            print_error "No active model: $ACTIVE_LINK does not exist"
            return 1
        fi
    fi
    print_status "models/active-model -> ${current:-<directory>}"
    if [[ ! -e "$ACTIVE_LINK" ]]; then
        print_error "BROKEN LINK: target does not exist"
        return 1
    fi
    target=$(cd "$ACTIVE_LINK" && pwd -P)
    print_status "resolved path: $target"
    validate_dir "$target" || true
    report_validation "${current:-active-model}"
    local prev; prev=$(previous_rel)
    [[ -n "$prev" ]] && print_status "previous model (for rollback): $prev"
    model_path="${MODEL_PATH:-$(read_env_key MODEL_PATH)}"
    if [[ -n "$model_path" && "${model_path%/}" != "models/active-model" && "${model_path%/}" != "$MODELS_DIR/active-model" ]]; then
        print_warning "MODEL_PATH=$model_path is set (env/.env): the worker loads THAT path, not models/active-model"
    fi
    if [[ -n "$(read_env_key LORA_MODEL_PATH)" ]]; then
        print_status "note: LORA_MODEL_PATH in .env is legacy and ignored by the worker"
    fi
    (( ${#MODEL_ERRORS[@]} == 0 ))
}

cmd_validate() {
    local arg="${1:-}"
    [[ -n "$arg" ]] || die "Usage: $(basename "$0") validate <name|path>"
    resolve_target "$arg" nocheck
    local dir="$TARGET_DIR"
    (( NEEDS_IMPORT == 1 )) && dir="$SOURCE_DIR"
    validate_dir "$dir" || true
    report_validation "$arg"
    (( ${#MODEL_ERRORS[@]} == 0 ))
}

# do_switch ARG ALLOW_IMPORT
do_switch() {
    local arg="$1" allow_import="${2:-1}" current check_dir
    resolve_target "$arg"
    if (( NEEDS_IMPORT == 1 && allow_import == 0 )); then
        die "Refusing: '$arg' resolves outside $MODELS_DIR"
    fi
    check_dir="$TARGET_DIR"
    (( NEEDS_IMPORT == 1 )) && check_dir="$SOURCE_DIR"
    if ! validate_dir "$check_dir"; then
        report_validation "$arg"
        die "Not switching: '$arg' is not a valid model. Nothing was changed."
    fi
    report_validation "$TARGET_REL"
    check_link_replaceable
    current=$(current_rel)
    if [[ "$current" == "$TARGET_REL" ]]; then
        print_success "$TARGET_REL is already the active model - nothing to do."
        return 0
    fi

    if (( DRY_RUN == 1 )); then
        print_header "Dry run - nothing will be changed"
        if (( NEEDS_IMPORT == 1 )); then
            if (( MOVE == 1 )); then
                print_status "[dry-run] would move (or copy, if on another filesystem) $SOURCE_DIR -> models/$TARGET_REL"
            else
                print_status "[dry-run] would copy $SOURCE_DIR -> models/$TARGET_REL"
            fi
        fi
        print_status "[dry-run] would repoint models/active-model: ${current:-<none>} -> $TARGET_REL"
        [[ -n "$current" ]] && print_status "[dry-run] would record '$current' as the previous model (for rollback)"
        restart_worker
        return 0
    fi

    if (( NEEDS_IMPORT == 1 )); then
        if ! confirm "Import $SOURCE_DIR into models/$TARGET_REL ($( (( MOVE == 1 )) && echo move || echo copy ))?"; then
            die "Cancelled - nothing was changed."
        fi
    fi
    with_lock    # only now - a rejected, cancelled or dry-run switch leaves no trace
    if (( NEEDS_IMPORT == 1 )); then
        import_model "$SOURCE_DIR" "$TARGET_REL"
        if ! validate_dir "$TARGET_DIR"; then
            report_validation "models/$TARGET_REL"
            die "The imported copy models/$TARGET_REL failed validation - active model NOT changed."
        fi
        print_success "Imported into models/$TARGET_REL"
    fi

    set_active "$TARGET_REL" "$current"
    print_success "models/active-model -> $TARGET_REL  (previous: ${current:-<none>})"
    restart_worker
    echo
    print_status "Verify:   $SELF current   and   $SCRIPTS_DIR/health_check.sh"
    [[ -n "$current" ]] && print_status "Rollback: $SELF rollback"
    return 0
}

cmd_rollback() {
    local prev
    prev=$(previous_rel)
    [[ -n "$prev" ]] || die "No previous model recorded ($PREVIOUS_FILE). Use '$(basename "$0") list' and 'switch <name>'."
    [[ "$prev" != /* && "$prev" != *..* ]] || die "Corrupt $PREVIOUS_FILE: '$prev'"
    [[ -d "$MODELS_DIR/$prev" ]] || die "Previous model models/$prev no longer exists"
    print_status "Rolling back to $prev"
    do_switch "$MODELS_DIR/$prev" 0
}

cmd_restart() {
    if (( DRY_RUN == 1 )); then
        print_status "[dry-run] would restart the AI worker"
        return 0
    fi
    NO_RESTART=0
    restart_worker
}

# ---------------------------------------------------------------------------
main() {
    local positional=()
    while (( $# > 0 )); do
        case "$1" in
            -n|--dry-run)  DRY_RUN=1 ;;
            -y|--yes)      ASSUME_YES=1 ;;
            --no-restart)  NO_RESTART=1 ;;
            --move)        MOVE=1 ;;
            --name)        shift; [[ $# -gt 0 ]] || die "--name needs a value"; IMPORT_NAME="$1" ;;
            --name=*)      IMPORT_NAME="${1#--name=}" ;;
            -h|--help)     show_help; return 0 ;;
            --)            shift; positional+=("$@"); break ;;
            -*)            print_error "Unknown option: $1"; show_help; return 1 ;;
            *)             positional+=("$1") ;;
        esac
        shift
    done
    export ASSUME_YES
    local cmd="${positional[0]:-list}"
    local arg="${positional[1]:-}"
    case "$cmd" in
        list|ls)        cmd_list ;;
        current|status) cmd_current ;;
        validate|check) cmd_validate "$arg" ;;
        switch|use)
            [[ -n "$arg" ]] || { print_error "Please specify the model to switch to"; show_help; return 1; }
            do_switch "$arg" 1 ;;
        rollback)       cmd_rollback ;;
        restart)        cmd_restart ;;
        help)           show_help ;;
        auto)
            die "'auto' was removed (it guessed version numbers). Use '$(basename "$0") list' then 'switch <name>'." ;;
        *)
            print_error "Unknown command: $cmd"
            show_help
            return 1 ;;
    esac
}

main "$@"

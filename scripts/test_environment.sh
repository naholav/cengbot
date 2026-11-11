#!/usr/bin/env bash
# =============================================================================
# CengBot environment test (read-only) - checks what training / inference need
# without running anything heavy. Exit code 1 if a required item is missing.
# =============================================================================
set -uo pipefail

# shellcheck source=scripts/_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

FAIL=0
fail() { print_error "$*"; FAIL=1; }

print_header "CengBot Environment Test ($PROJECT_ROOT)"

for f in src/train_model.py src/ai_model_worker.py src/llama_model_handler.py; do
    if [[ -f "$PROJECT_ROOT/$f" ]]; then print_success "$f found"; else fail "$f not found"; fi
done

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    print_success ".env found"
    for key in TELEGRAM_BOT_TOKEN HUGGING_FACE_TOKEN; do
        if [[ -n "$(read_env_key "$key")" ]]; then
            print_success "$key is set (value not shown)"
        else
            print_warning "$key is empty or missing in .env"
        fi
    done
else
    fail ".env not found (copy .env.example to .env)"
fi

if [[ -x "$PYTHON" ]]; then
    print_success "Python: $("$PYTHON" --version 2>&1) ($PYTHON)"
    for mod in torch transformers peft; do
        if ver=$("$PYTHON" -c "import $mod; print($mod.__version__)" 2>/dev/null); then
            print_success "$mod $ver"
        else
            fail "$mod not importable with $PYTHON"
        fi
    done
    if "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        print_success "CUDA available to torch"
    else
        print_warning "torch reports no CUDA device (training/inference would run on CPU)"
    fi
else
    fail "Project Python not found: $PYTHON"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    print_success "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1)"
else
    print_warning "nvidia-smi not found - GPU may not be available"
fi

if compgen -G "$PROJECT_ROOT/model_cache/models--meta-llama--Llama-3.2-3B/snapshots/*/config.json" >/dev/null; then
    print_success "Base model cached in model_cache/ (offline loading possible)"
else
    print_warning "Base model not cached in model_cache/ - first load needs network + HUGGING_FACE_TOKEN"
fi

found=0
for ds in data/merged_qa_data.jsonl data/cengbot_qa_augmented.jsonl; do
    if [[ -f "$PROJECT_ROOT/$ds" ]]; then
        print_success "Dataset $ds: $(wc -l < "$PROJECT_ROOT/$ds") lines"
        found=1
    fi
done
(( found == 1 )) || print_warning "No training dataset in data/ (needed only for training)"

if "$SCRIPTS_DIR/switch_model.sh" validate "$PROJECT_ROOT/models/active-model" >/dev/null 2>&1; then
    print_success "Active model: models/active-model -> $(readlink "$PROJECT_ROOT/models/active-model") (valid)"
else
    fail "models/active-model is missing or invalid (scripts/switch_model.sh current)"
fi

print_header "Environment Test Complete"
if (( FAIL == 0 )); then
    print_success "Environment looks good. Training: ./scripts/train_model.sh --dry-run"
else
    print_error "Some required items are missing (see above)"
fi
exit "$FAIL"

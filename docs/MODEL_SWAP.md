# Model değiştirme / Model swap runbook

This runbook takes a newly trained model from training to production and back.
Every command and message in it comes from `scripts/train_model.sh`,
`src/train_model.py` and `scripts/switch_model.sh`.

**How the model is chosen.** The AI worker (`src/ai_model_worker.py`) loads the
model behind the symlink **`models/active-model`** once, at start-up. That
symlink is the single source of truth, and `scripts/switch_model.sh` is the only
tool that should change it. Switching a model means repointing the symlink and
restarting **only the AI worker**. The Telegram bot and the Admin API keep
running.

> **`.env` settings that affect the model**
>
> - `MODEL_PATH` is an *optional* override (default `models/active-model`,
>   relative to the project root). If it is set, the worker loads that path
>   instead of the symlink, and `switch_model.sh` has no effect until you remove
>   it (`switch_model.sh current` warns about it). Normally leave it empty.
> - `LORA_MODEL_PATH` is **legacy and ignored**. The worker logs a warning if it
>   points somewhere else; delete the line.
> - Loading is **offline-first**. The base model named in the adapter
>   (today `meta-llama/Llama-3.2-3B`) comes from `model_cache/` (or the default
>   Hugging Face cache) with `local_files_only=True`; huggingface.co is
>   contacted only if files are missing and downloads are allowed. In
>   production set `MODEL_ALLOW_DOWNLOAD=false` or `HF_HUB_OFFLINE=1`: a model
>   whose base is not cached then fails to load instead of downloading, and
>   `validate` warns about this in advance.
> - `MODEL_LOAD_IN_4BIT=true` loads the base model in 4-bit NF4 (bitsandbytes,
>   CUDA only) for bases that do not fit in bf16, about 12B and larger on a
>   24 GB GPU. It is slower, and the LoRA adapter then stays unmerged. Default
>   `false`.
> - `MODEL_TRUST_REMOTE_CODE=true` **executes Python code from the model
>   repository** (training and inference). Only legacy remote-code models such
>   as `THUDM/chatglm3-6b` or `internlm/internlm2_5-7b-chat` need it; leave it
>   `false` (the default) for every family in the table below.
> - `LORA_TARGET_MODULES` (training only; `--lora-target-modules` wins):
>   `auto` (default) adapts `q/k/v/o/gate/up/down_proj` when the model has all
>   of them, otherwise every linear layer (`all-linear`, for the fused
>   projections of Phi-3/4 and GLM); `all-linear`; or a comma-separated list of
>   module names.

All commands run from the project root. The scripts find the project from
their own location, so they also work from any other directory.

---

## 0. Before you start

```bash
./scripts/switch_model.sh current     # note the active model: this is your rollback target
./scripts/test_environment.sh         # packages, CUDA, base model cache, datasets, active model (read-only)
./scripts/backup_data.sh              # optional: DB snapshot + data/ + models/ -> backups/ (--no-models to skip models)
```

Check that `MODEL_PATH` is empty in `.env` (see above).

## 1. Yeni modeli eğit / Train the new model

```bash
./scripts/export_training_data.sh           # optional: DB -> data/cengbot_qa_augmented.jsonl
./scripts/augment_training_data.sh --dry-run  # optional: Claude API augmentation (costs money; see README)
./scripts/train_model.sh --dry-run          # data + tokenization check only: no model, nothing is written
./scripts/train_model.sh                    # GPU training (hours); asks first (--yes to skip the question)
```

- **Training data.** By default `src/train_model.py` reads
  `data/merged_qa_data.jsonl` + `data/cengbot_qa_augmented.jsonl`, and before a
  real run it refreshes `data/cengbot_qa_augmented.jsonl` from the approved
  database rows itself (skip that with `--no-db-export`). Other files:
  `--dataset FILE...` or `TRAINING_DATASET` in `.env`.
- **Options.** Every option that `train_model.sh` does not know is passed to
  `src/train_model.py`, for example `--epochs 3`, `--lr 1e-4`,
  `--load-in-4bit`, `--semantic-dedup`, `--output-name NAME`. See
  `.venv/bin/python src/train_model.py --help` for the full list.
- **Base model.** `--base-model ID|DIR` (default `BASE_MODEL_NAME` from `.env`,
  else `meta-llama/Llama-3.2-3B`). For another family see
  [Retrain on a different base model](#retrain-on-a-different-base-model).
- **GPU memory.** If the AI worker is running, `train_model.sh` warns that it
  shares the GPU with training. If the GPU cannot hold both, stop the worker
  first (systemd: `sudo systemctl stop cengbot-worker`; manual mode:
  `./scripts/stop_system.sh` stops all services) and start it again after
  training. While the worker is stopped, users get the bot's fallback reply.
- **Output directory.** The new model is written to
  **`models/final-best-model-v<N>/`**, where `<N>` is the highest integer
  version in `models/` plus one: with v1, v1.1 and v1.2 present, the next one is
  **v2**. The directory is written to a hidden temporary directory first and
  renamed into place, so a crash never leaves a half-written model. It contains
  `adapter_config.json`, `adapter_model.safetensors`, the tokenizer files and
  `training_info.json` (no `method1/`, no optimizer state). An existing
  directory is never overwritten unless you pass `--overwrite`, and the active
  model is never overwritten at all.
- **Checkpoints** go to `checkpoints/final-best-model-v<N>/`. An interrupted
  run can be resumed with `./scripts/train_model.sh --resume-from-checkpoint latest`.
- **Logs.** `logs/training/auto_training_<timestamp>.log` (wrapper output),
  `logs/training_history/v<K>.log` + `v<K>_info.json` and
  `logs/training_<timestamp>.log`. `<K>` is the *training session* number,
  which is counted separately from the model version `<N>`;
  `./scripts/view_training_history.sh list|view vK|compare vA vB|best` reads it.
- **End of a successful run.** The last lines are `FINAL_MODEL_DIR=.../models/final-best-model-v2`,
  then `train_model.sh` validates the directory and prints
  `The new model satisfies the model contract.` and the exact `switch` command.

**Training never changes the active model.**

## 2. A model from somewhere else

**TR:** `scripts/train_model.sh` ile eğitilen model otomatik olarak
`models/final-best-model-v<N>/` dizinine yazılır. Başka bir yerden gelen modeli
`models/<isim>/` altına kopyalayın ya da 4. adımda dışarıdaki yolu doğrudan
verin; script modeli doğrular ve onay isteyerek `models/` içine kopyalar.

**EN:** A model trained elsewhere can be copied to `models/<name>/`, or you can
pass its outside path to `switch` in step 4. The script validates it and, after
asking, copies it into `models/<name>` (`--name NAME` to choose the name,
`--move` to move it instead when it is on the same filesystem).

A valid model directory is **one of**:

| Type | Required files |
|---|---|
| LoRA adapter (normal case) | `adapter_config.json` + `adapter_model.safetensors`, directly in the directory or in `method1/` (the legacy layout of `final-best-model-v1`). Optional: tokenizer files, `training_info.json`. The base model comes from `base_model_name_or_path` in `adapter_config.json` (fallback `BASE_MODEL_NAME`). |
| Full / merged model | `config.json` + `*.safetensors` (or `pytorch_model*.bin`) + tokenizer files (`tokenizer.json`, `tokenizer.model`, or `vocab.json` + `merges.txt`) |

Any architecture that the installed transformers supports natively works
(Llama, Qwen, DeepSeek distills, Mistral, Gemma, Phi, GLM, ...). `validate`
also checks that.

`adapter_model.bin` alone is **not** accepted by `switch_model.sh`. Re-save the
adapter with `save_pretrained(..., safe_serialization=True)`.

## 3. Doğrula / Validate

```bash
./scripts/switch_model.sh list                          # all models: type, size, loss/steps/date; marks ACTIVE and (previous)
./scripts/switch_model.sh validate final-best-model-v2  # or: validate v2 / validate /path/to/model
```

`validate` checks the files above, that the JSON files parse, that every
safetensors header fits its file (this catches truncated or half-copied files),
whether the base model is cached in `model_cache/` and whether the installed
transformers supports its architecture. It reads nothing big and
changes nothing. Exit code 0 means valid; warnings (for example "base model ...
is not cached") do not make it fail.

Optional: see exactly what the worker *would* load. This only looks at the
filesystem; no weights are loaded:

```bash
MODEL_PATH=models/final-best-model-v2 .venv/bin/python -c \
  "import sys; sys.path.insert(0,'src'); from llama_model_handler import resolve_model_spec; print(resolve_model_spec())"
```

## 4. Geçiş yap / Switch

```bash
./scripts/switch_model.sh switch final-best-model-v2 --dry-run   # show the plan, change nothing
./scripts/switch_model.sh switch final-best-model-v2             # validate + switch + restart the worker (asks)
./scripts/switch_model.sh switch final-best-model-v2 --yes       # same, no questions
```

`v2` and `2` are shortcuts for `final-best-model-v2`. What `switch` does:

1. Validates the model. If it is invalid, **nothing is changed**.
2. If the model is already active, it prints `... is already the active model - nothing to do.`
   and stops (no restart; use `restart` if you changed the files in place).
3. For a path outside `models/`, copies it to `models/<name>` after asking.
   It refuses to overwrite an existing `models/<name>`.
4. Atomically repoints `models/active-model` to a relative target (`rename(2)`,
   so there is never a moment without a link).
5. Records the old model in `models/.previous-model` and appends a line to
   `models/.switch-history.log`. A lock (`models/.switch.lock`) prevents two
   switches at the same time.
6. Restarts **only the AI worker**:
   - systemd mode (`cengbot-worker.service` is active for this project):
     `sudo systemctl restart cengbot-worker` (sudo may ask for your password).
   - manual mode (`start_system.sh`): asks, then stops this project's worker
     (SIGTERM, up to 60 s, then SIGKILL) and starts it again, appending to
     `logs/worker.log`. Without a terminal the question counts as "no": pass `--yes`.
   - worker not running: prints the start command; the new model is loaded when it starts.
   - `--no-restart` (or `CENGBOT_NO_RESTART=1`): only prints the command. Restart
     later with `./scripts/switch_model.sh restart --yes`.

Models are never deleted.

**During the restart** the worker needs a few minutes to load the model.
Questions asked meanwhile are not lost: they wait in RabbitMQ and are answered
and stored in the database once the model is loaded. The users who asked them,
however, get the bot's fallback reply in Telegram, because the bot waits only
`BOT_ANSWER_TIMEOUT` seconds (default 30) for an answer. Switch at a quiet time.

## 5. Kontrol et / Verify

```bash
./scripts/switch_model.sh current          # symlink target + validation (exit code 1 if invalid)
tail -f logs/worker.log                    # in systemd mode the units also write to this file
#   expect: "Loading lora model .../models/final-best-model-v2 (base=<base model>, cached=True, ...)"
#           "Model final-best-model-v2 loaded in ...s"
#           "Connected to RabbitMQ (questions=questions, answers=answers)"
#           "Waiting for questions on 'questions'..."
./scripts/health_check.sh                  # services, /health, active model valid, disk/GPU, log sizes
curl -s http://localhost:8001/health       # public status: database + model path available
```

**Test question:** send a question to the bot in the Telegram group, for
example "Bilgisayar mühendisliği bölüm başkanı kimdir?" and "What are the
graduation requirements?". Check that the answer arrives and makes sense in both
Turkish and English. The question and answer also appear in the admin panel.

Every answer stores the name of the model that produced it in
`raw_data.model_version`, so you can confirm which model answered:

```bash
.venv/bin/python -c "import sqlite3; c=sqlite3.connect('university_bot.db'); print(c.execute('SELECT id, model_version, answered_at FROM raw_data ORDER BY id DESC LIMIT 5').fetchall())"
```

**Benchmark.** `.venv/bin/python benchmark/run_benchmark.py --label <name>`
scores the active model on the 100-question benchmark and adds it to
[`benchmark/LEADERBOARD.md`](../benchmark/LEADERBOARD.md); see step 9 of
[Retrain on a different base model](#retrain-on-a-different-base-model).

## 6. Geri al / Rollback

```bash
./scripts/switch_model.sh rollback         # back to models/.previous-model (validates, then restarts the worker; asks)
./scripts/switch_model.sh rollback --yes
./scripts/switch_model.sh switch final-best-model-v1.2   # or go to any specific model
```

`rollback` swaps the current and the previous model, so running it twice brings
you back. It only accepts models inside `models/`.

## Retrain on a different base model

The production model is Llama-3.2-3B + LoRA, but nothing in training,
validation, switching or inference is tied to Llama. Train a LoRA adapter on
another open-weight base and swap it in like any other model; the adapter
records its base (`base_model_name_or_path`), so the worker loads the right
one. Family details are handled in code: the stop tokens of each family, BOS
handling, `<think>` blocks of reasoning models (removed from answers), LoRA
target modules (`auto`) and eager attention for Gemma training. The prompt is
the same plain `Öğrenci:` / `Student:` template for every family (no chat
template), so instruct and base checkpoints both work.

```bash
./scripts/switch_model.sh current                      # 1. note the rollback target

# 2. Gated base (Llama, Gemma, most Mistral): accept the licence on huggingface.co and set
#    HUGGING_FACE_TOKEN in .env. The first run downloads the base into model_cache/ once
#    (MODEL_ALLOW_DOWNLOAD not false, HF_HUB_OFFLINE unset). Optional: BASE_MODEL_NAME=<base> in .env.
#    Legacy remote-code repositories only: MODEL_TRUST_REMOTE_CODE=true in .env (training and worker).
BASE=Qwen/Qwen2.5-7B-Instruct

./scripts/train_model.sh --dry-run --base-model "$BASE"   # 3. data + tokenization check (fetches only the tokenizer)
sudo systemctl stop cengbot-worker                        # 4. 7B+: free the GPU (manual mode: ./scripts/stop_system.sh)
./scripts/train_model.sh --yes --base-model "$BASE" --load-in-4bit   # 5. train; <= 4B: drop --load-in-4bit
./scripts/switch_model.sh validate final-best-model-v2   # 6. validate

# 7. Base of ~12B or more: MODEL_LOAD_IN_4BIT=true in .env (4-bit inference).

./scripts/switch_model.sh switch final-best-model-v2 --yes   # 8. switch ...
sudo systemctl start cengbot-worker                          #    ... start the worker again if you stopped it (manual mode: ./scripts/start_system.sh)
./scripts/switch_model.sh current; tail -f logs/worker.log   #    ... verify (section 5)
#    bad answers? ./scripts/switch_model.sh rollback --yes

.venv/bin/python benchmark/run_benchmark.py --label v2-qwen2.5-7b   # 9. score the new model
```

**Step 9, benchmark.** Score every new model on the fixed 100-question
benchmark and compare it with the earlier runs on
[`benchmark/LEADERBOARD.md`](../benchmark/LEADERBOARD.md) (current baseline:
Llama-3.2-3B + LoRA v1.2, 30.2/100 deterministic). The benchmark loads its own
copy of the model: a 3-4B model fits next to the running worker; for a 7B+
model run it before step 8, while the worker is still stopped, with
`--model-path models/final-best-model-v2`, which also shows the score before any
user sees the new model. Before retraining, refresh the links in the training
data: the v1.2 answers often point to wrong or dead pages
([README: Benchmark](../README.md#benchmark), [benchmark/README.md](../benchmark/README.md)).

### Families

"Gated" means: accept the licence on the model page and set
`HUGGING_FACE_TOKEN` before the first download. Licences change, so check the
model page.

| Family | Example Hugging Face ids | On a 24 GB GPU (RTX 3090) | Gated | Notes |
|---|---|---|---|---|
| Llama 3.x | `meta-llama/Llama-3.2-3B` (current), `meta-llama/Llama-3.1-8B-Instruct` | 3B: bf16. 8B: train with `--load-in-4bit`, inference bf16 (~16 GB) | yes | explicit LoRA targets |
| Qwen2.5 / Qwen3 | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-14B-Instruct` | ≤ 4B: bf16. 7-8B: QLoRA + bf16 inference. 14B: QLoRA + `MODEL_LOAD_IN_4BIT=true` | no | no BOS token (handled); `<think>` blocks removed |
| DeepSeek-R1 distills | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | QLoRA + bf16 inference | no | Qwen2 / Llama architecture; a reasoning model, not ideal for short answers |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3`, `mistralai/Mistral-Nemo-Instruct-2407` (12B) | 7B: QLoRA + bf16 inference. 12B: QLoRA + 4-bit inference | most | |
| Gemma 2 / 3 | `google/gemma-2-2b-it`, `google/gemma-2-9b-it`, `google/gemma-3-1b-it` | 2B: bf16. 9B: QLoRA; inference bf16 (~19 GB, tight) or 4-bit | yes | BOS required (handled); eager attention for training; large vocabulary: `--batch-size 2 --grad-accum 32`; Gemma 3 4B+ (multimodal) untested |
| Phi-3.5 / Phi-4 | `microsoft/Phi-3.5-mini-instruct`, `microsoft/Phi-4-mini-instruct`, `microsoft/phi-4` (14B) | mini: bf16. 14B: QLoRA + 4-bit inference | no | fused projections: `auto` uses all-linear |
| GLM-4 | `THUDM/glm-4-9b-chat-hf`, `THUDM/GLM-4-9B-0414` | QLoRA; inference bf16 (~19 GB) or 4-bit | check | all-linear; the non-`-hf` `glm-4-9b-chat` and `chatglm3` need remote code |
| Granite 3.x | `ibm-granite/granite-3.3-2b-instruct`, `ibm-granite/granite-3.3-8b-instruct` | 2B: bf16. 8B: QLoRA + bf16 inference | no | |
| OLMo 2 | `allenai/OLMo-2-0425-1B-Instruct`, `allenai/OLMo-2-1124-7B-Instruct` | 1B: bf16. 7B: QLoRA + bf16 inference | no | |
| Falcon3 | `tiiuae/Falcon3-3B-Instruct`, `tiiuae/Falcon3-7B-Instruct`, `tiiuae/Falcon3-10B-Instruct` | 3B: bf16. 7-10B: QLoRA; 10B: 4-bit inference | no | Llama architecture |
| SmolLM2 / 3 | `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `HuggingFaceTB/SmolLM3-3B` | bf16 | no | weak Turkish |
| Legacy remote code | `THUDM/chatglm3-6b`, `internlm/internlm2_5-7b-chat` | `--trust-remote-code` + `MODEL_TRUST_REMOTE_CODE=true` | - | executes repository code: not recommended |

Rule of thumb: bf16 inference fits up to about 9-10B parameters; from about
12B set `MODEL_LOAD_IN_4BIT=true`. Stop the worker while training a 7B+ base.
For reference, a short QLoRA run on Llama-3.2-3B peaked at 4.5 GiB of GPU
memory, and 4-bit inference of the same model at 2.3 GiB.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NOT a valid model directory` | The lines below it list the missing or broken files. Copy the model again. |
| `base model ... is not cached in model_cache/` (warning) | The first load or training run downloads the base once: it needs network access, downloads must be allowed (`MODEL_ALLOW_DOWNLOAD` not `false`, `HF_HUB_OFFLINE` unset), and a gated base (Llama, Gemma, most Mistral) also needs `HUGGING_FACE_TOKEN` with the licence accepted. Afterwards everything runs offline. |
| Base not cached, but the server must stay offline (any family, e.g. a Qwen or Phi model) | Download it once on a machine with network access, for example `.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='model_cache')"`, and copy the directory `model_cache/models--<org>--<name>` into the server's `model_cache/`. Or switch to a model whose base is already cached. |
| `architecture '...' is not supported by the installed transformers` (warning), or the worker logs `this transformers version does not support the architecture` | The base is newer than the installed transformers (`config/requirements.txt` pins `<5`). Use a natively supported model from the table above, or upgrade transformers and test on the GPU before switching. |
| `ships custom code (auto_map): loading it needs MODEL_TRUST_REMOTE_CODE=true` (warning), or the worker logs `needs custom code from its repository` | The repository needs remote code. Prefer a checkpoint that is native in transformers (for GLM the `-hf` or `-0414` variant). Only for a repository you trust: `MODEL_TRUST_REMOTE_CODE=true` in `.env` (and `--trust-remote-code` for training); this executes the repository's Python code. |
| `CUDA out of memory` while training | Stop the worker first, train with `--load-in-4bit`, and/or use `--batch-size 2 --grad-accum 32` (same effective batch). |
| `The AI worker exited right after starting` | Check `logs/worker.log`, then `./scripts/switch_model.sh rollback --yes`. |
| Worker keeps logging `Model load failed ... Retrying in Ns` | The worker retries with exponential backoff (5 s up to 300 s) instead of exiting. Fix the cause shown in the log, or roll back. |
| `models/active-model exists and is NOT a symlink` | Someone replaced the link with a directory. Move it away (`mv models/active-model models/<name>`), then `switch <name>`. |
| `Another switch_model.sh is running` | Wait for the other run to finish. The lock (`flock` on `models/.switch.lock`) is released automatically when that process exits. |
| Old model still answers | The worker was not restarted: `./scripts/switch_model.sh restart --yes`. Also check that `MODEL_PATH` is not set in `.env`. |
| `Model directory already exists` (training) | Pass `--output-name NEW_NAME`, or `--overwrite` to replace a non-active model. |

## Command reference

```
scripts/switch_model.sh [options] <command> [model]
```

| Command (alias) | What it does |
|---|---|
| `list` (`ls`, also the default) | Models in `models/` with type, size and training info; marks the active and the previous one |
| `current` (`status`) | Active model, its validation, the rollback target; warns about `MODEL_PATH` / `LORA_MODEL_PATH` |
| `validate <name\|path>` (`check`) | Check a model directory against the contract above |
| `switch <name\|path>` (`use`) | Validate, import if outside `models/`, repoint `models/active-model`, restart the worker |
| `rollback` | Switch back to `models/.previous-model` |
| `restart` | Restart only the AI worker |
| `help` | Show the help |

| Option | Meaning |
|---|---|
| `-n`, `--dry-run` | Show what would happen; change nothing |
| `-y`, `--yes` | Do not ask for confirmation (import, worker restart) |
| `--no-restart` | Do not restart the worker; print the command instead (same as `CENGBOT_NO_RESTART=1`) |
| `--name NAME` | Directory name under `models/` when importing an outside path |
| `--move` | Move instead of copy an outside model (same filesystem only; otherwise it is copied) |

Environment: `CENGBOT_ROOT` (project root), `CENGBOT_PYTHON` (interpreter used
for validation), `CENGBOT_NO_RESTART=1`; `BASE_MODEL_NAME`, `MODEL_CACHE_DIR`
and `MODEL_TRUST_REMOTE_CODE` are read from the environment or `.env`.

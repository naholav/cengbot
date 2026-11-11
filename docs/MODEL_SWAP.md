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
> - Loading is **offline-first**. The base model (`meta-llama/Llama-3.2-3B`)
>   comes from `model_cache/` (or the default Hugging Face cache) with
>   `local_files_only=True`; huggingface.co is contacted only if files are
>   missing and downloads are allowed. In production set
>   `MODEL_ALLOW_DOWNLOAD=false` or `HF_HUB_OFFLINE=1`: a model whose base is not
>   cached then fails to load instead of downloading, and `validate` warns about
>   this in advance.

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
| Full / merged model | `config.json` + `*.safetensors` (or `pytorch_model*.bin`) + `tokenizer.json` or `tokenizer.model` |

`adapter_model.bin` alone is **not** accepted by `switch_model.sh`. Re-save the
adapter with `save_pretrained(..., safe_serialization=True)`.

## 3. Doğrula / Validate

```bash
./scripts/switch_model.sh list                          # all models: type, size, loss/steps/date; marks ACTIVE and (previous)
./scripts/switch_model.sh validate final-best-model-v2  # or: validate v2 / validate /path/to/model
```

`validate` checks the files above, that the JSON files parse, that every
safetensors header fits its file (this catches truncated or half-copied files)
and whether the base model is cached in `model_cache/`. It reads nothing big and
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
#   expect: "Loading lora model .../models/final-best-model-v2 (base=meta-llama/Llama-3.2-3B, cached=True, ...)"
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

## 6. Geri al / Rollback

```bash
./scripts/switch_model.sh rollback         # back to models/.previous-model (validates, then restarts the worker; asks)
./scripts/switch_model.sh rollback --yes
./scripts/switch_model.sh switch final-best-model-v1.2   # or go to any specific model
```

`rollback` swaps the current and the previous model, so running it twice brings
you back. It only accepts models inside `models/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `NOT a valid model directory` | The lines below it list the missing or broken files. Copy the model again. |
| `base model ... is not cached in model_cache/` (warning) | Loading needs network access plus `HUGGING_FACE_TOKEN` with access to the gated Llama model, and is impossible with `MODEL_ALLOW_DOWNLOAD=false` / `HF_HUB_OFFLINE=1`. Download the base model once, or use a model trained on the cached `meta-llama/Llama-3.2-3B`. |
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
for validation), `CENGBOT_NO_RESTART=1`.

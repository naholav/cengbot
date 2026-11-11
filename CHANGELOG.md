# Changelog

All notable changes to CengBot are listed here.

## 2026-11-11 - Security, reliability and model-swap release

A hardening release: no more hardcoded credentials, an authenticated admin API,
a worker that cannot flood the disk, a reproducible training pipeline whose
prompts match inference, and a one-command model swap for the next retrain.

### Security

- **Hardcoded passwords removed.** The admin API no longer falls back to a
  built-in admin password, and `start_system.sh` no longer pipes a stored sudo
  password into `sudo -S`. Admin credentials come only from `ADMIN_USERNAME`
  plus `ADMIN_PASSWORD_HASH` (SHA-256 hex) or `ADMIN_PASSWORD`; if neither is
  set, every login is rejected and an error is logged. Scripts that need root
  use interactive `sudo`.
- **All admin data endpoints now require authentication.** Previously only the
  document viewer and the login checked credentials. Public are only `GET /`, a
  minimal `GET /health` (system and GPU details need credentials) and the OpenAPI docs.
- **`/docs/read` can no longer read `.env`.** It serves only an allowlist
  (Markdown files, `logs/*.log`, `scripts/*.sh`), rejects path traversal and
  returns at most 1 MiB (the tail of a log), instead of any file of any size.
- **Prompt guard** (`src/prompt_guard.py`): jailbreak and prompt-injection
  attempts in Turkish and English, and over-long questions, are refused before
  they reach the model; answers that leak the system prompt are replaced.
- Safer panel credential storage: the Basic auth header is kept in memory and
  `sessionStorage` instead of `localStorage`, and the panel logs out on 401.
  Production builds ship without source maps.
- The bot token no longer appears in the logs (per-request HTTP logging is
  silenced), and user and model text is HTML-escaped in Telegram replies.
- Vote buttons are validated: well-formed callback data from the allowed chat,
  on the bot's own reply message only.
- Models are loaded without `trust_remote_code`; CORS origins come from
  `CORS_ORIGINS` instead of a hardcoded list; database errors return a generic
  message; the deploy script keeps `.env` at mode 600 instead of making the
  project world-readable.
- Dependency minimums raised to audited versions (`config/requirements.txt`).

### Reliability

- **The worker no longer busy-loops when the model or RabbitMQ is
  unavailable.** It loads the model before connecting and retries with
  exponential backoff (5 s up to 300 s) and one concise log line per attempt.
  Previously a failed model load restarted the consumer immediately, in a loop
  that filled the disk with logs.
- **Poison messages are bounded.** A failing question is retried with an
  attempt counter and, after 3 attempts (`WORKER_MAX_ATTEMPTS`), answered with
  an apology and acknowledged, instead of `basic_nack(requeue=True)` forever.
- **Log rotation everywhere.** `logs/error.log` is size-rotated by Python
  (10 MB × 5) and repeated identical errors are collapsed; the service logs are
  rotated by logrotate (`config/logrotate/cengbot.conf`, via
  `scripts/rotate_logs.sh` or the system-wide install), with a start-time
  safety net for oversized logs. `src/system_monitor.py --check-logs` reports
  log sizes and free disk space.
- Generation runs in a helper thread so AMQP heartbeats keep flowing; queues
  are durable, messages persistent, publishes confirmed.
- Every process resolves the same SQLite file (relative `DATABASE_URL` paths
  are resolved against the project root).

### Model and training

- **Offline-first model loading.** The base model is loaded from
  `model_cache/` with `local_files_only`; downloads happen only when files are
  missing and can be forbidden with `MODEL_ALLOW_DOWNLOAD=false` or `HF_HUB_OFFLINE=1`.
- **`models/active-model` is the single source of truth.** `MODEL_PATH` is an
  optional override and `LORA_MODEL_PATH` is ignored. `scripts/switch_model.sh`
  (`list`, `current`, `validate`, `switch`, `rollback`, `restart`) validates a
  model against the directory contract, repoints the symlink atomically,
  remembers the previous model and restarts only the worker. The guessing
  `auto` mode was removed. Runbook: `docs/MODEL_SWAP.md`.
- **Train/inference prompt alignment.** Inference now uses the exact template
  the adapter was trained with, and `src/train_model.py` imports the same
  system prompts, so the training prompt is byte-identical to the inference
  prompt (the old long inference prompt is still available as
  `MODEL_PROMPT_STYLE=legacy`).
- **Correct label masking with EOS.** The loss covers the answer tokens and the
  EOS token only, so the model learns to stop; over-long examples are dropped or
  truncated and always reported.
- **Group-aware train/validation split.** An original question, its augmented
  variants, exact duplicates and rows about the same course code stay on the
  same side, so validation loss is not inflated by leakage.
- Training writes `models/final-best-model-v<N>/` atomically (adapter,
  tokenizer, `training_info.json`; no `method1/`, no optimizer state), keeps
  checkpoints in `checkpoints/`, supports `--dry-run` and
  `--resume-from-checkpoint latest`, and never changes the active model.
- Faster inference: the LoRA weights are merged into the base model at load
  time (`MODEL_MERGE_LORA`); answers stop at turn markers and after 4 sentences.
- Data export (`src/database_to_training.py`) merges into the existing file and
  tags rows with a `group_id`; augmentation (`src/data_augmentation.py`)
  defaults to `claude-sonnet-5-5`, enables the server-side refusal fallback by
  default (`CLAUDE_REFUSAL_FALLBACK`), supports `AUGMENT_CLAUDE_EFFORT`, and adds
  `--dry-run`, `--max-items`, a confirmation and a resumable checkpoint.

### Bot

- **Concurrent bot with a per-question answer router.** Updates are handled
  concurrently and each answer is delivered to the handler waiting for that
  question, instead of every handler polling one shared queue. One slow answer
  no longer blocks other users.
- **Vote/like sync.** `raw_data.like` now follows the per-user votes, so the
  panel's statistics show the users' feedback; one vote per user and question
  (at most two changes), safe against double clicks.
- Long answers are split to fit Telegram's limit; the answer timeout
  (`BOT_ANSWER_TIMEOUT`) and the allowed chat (`TELEGRAM_ALLOWED_CHAT_ID`) are
  configurable; duplicate detection runs in the background; RabbitMQ and
  Telegram connection errors are retried with backoff.

### Admin API and panel

- Router-level authentication, no default password, constant-time comparison;
  the startup log says how authentication is configured and warns about a
  non-ASCII password (HTTP Basic credentials are decoded as ASCII).
- Stable, bounded pagination (`page_size` ≤ 100) with batched vote statistics.
- Editing an answer also updates its approved training copy; automatic
  refusal/apology answers cannot be approved until edited; deleting a question
  also removes its training copy, votes and duplicate references;
  `POST /detect-duplicates` recomputes everything in one transaction.
- `/health` returns `503` when the database is unreachable and reads GPU data
  from `nvidia-smi` (the API no longer initialises CUDA).
- Panel: server-side "pending only" filter, clearer error messages,
  `REACT_APP_API_URL` / `REACT_APP_ADMIN_USERNAME` build settings.

### Scripts and ops

- Shared `scripts/_common.sh`: project-relative paths, the project venv,
  `--help` / `--dry-run` / `--yes`, and process handling that only ever touches
  this project's processes.
- `start_system.sh` / `stop_system.sh` / `health_check.sh` rewritten; systemd
  units (`cengbot-worker`, `-bot`, `-admin`, `-monitor`) and logrotate installed
  by `optimize_linux.sh` (harmful kernel and GPU tweaks removed);
  `deploy_production.sh` sets up units, nginx, a health cron and an optional firewall.
- Rewritten or newly added: `switch_model.sh`, `train_model.sh` (wrapper with
  model validation), `rotate_logs.sh`, `backup_data.sh` (consistent SQLite snapshot),
  `export_training_data.sh`, `export_database.sh`, `augment_training_data.sh`,
  `view_training_history.sh`, `test_environment.sh`, `cleanup_system.sh` (safe
  defaults). `dev_mode.sh` was removed.
- `.env.example` documents every variable; documentation rewritten to match
  the code.

### Testing

- About 940 offline pytest tests (`.venv/bin/python -m pytest`) covering the
  API, worker, bot, model handler, prompt guard, training pipeline, database
  and scripts. They use a temporary database and log directory and never touch
  Telegram, RabbitMQ, Hugging Face or the Claude API. The GPU test is opt-in
  (`RUN_GPU_TESTS=1`); shellcheck runs when available (`SHELLCHECK=`).
- Frontend tests with Jest and Testing Library:
  `cd admin_frontend && CI=true npx react-scripts test --watchAll=false`.

### Upgrade notes

1. Set `ADMIN_PASSWORD_HASH` (or `ADMIN_PASSWORD`) in `.env`; there is no default password any more.
2. Remove `LORA_MODEL_PATH`, leave `MODEL_PATH` empty, and set `API_HOST=127.0.0.1`,
   `CORS_ORIGINS` and `MODEL_ALLOW_DOWNLOAD=false` (see `.env.example`).
3. `.venv/bin/python -m pip install -r config/requirements.txt -r config/requirements-dev.txt`
4. Install log rotation: `./scripts/rotate_logs.sh --install-cron` or
   `sudo ./scripts/optimize_linux.sh --only logrotate`.
5. Backfill `raw_data.like` for votes cast before this release (once):
   `.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from database_models import SessionLocal, sync_like_from_votes; s = SessionLocal(); print(sync_like_from_votes(s)); s.close()"`
6. Rebuild the admin panel; admins have to log in again.

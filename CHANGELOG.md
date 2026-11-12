# Changelog

All notable changes to CengBot are listed here.

## 2026-11-12 - Verified and faster training, WAL database, late answers, det-v3 benchmark

### Dependencies

- ML stack upgraded after GPU and benchmark checks: torch 2.15.0 (CUDA 13.2
  wheels), transformers 4.57.6, accelerate 1.15.0, peft 0.21.2, bitsandbytes
  0.50.2. Closes the torch `weights_only` unpickler bypass (<2.10) and the
  accelerate sharded-checkpoint path traversal (<1.15). Same answer quality
  and speed on v1.2 (benchmark 30.2 vs 31.4 det-v2, 36.9 vs 36.5 tok/s; all
  CPU, GPU and live tests pass). Models load with `dtype=` (the old
  `torch_dtype=` is deprecated). Minimums in `config/requirements.txt` follow
  the validated versions.

### Training

- Verified end to end on the GPU (export, dry run on 22k examples, bf16 and
  4-bit training of Llama-3.2-3B, validation, loading, benchmark smoke run);
  opt-in GPU test `tests/test_training_e2e_gpu.py` (`RUN_GPU_TESTS=1`).
- `train_model.py --max-steps N` (smoke runs; evaluation and checkpoints are
  spread over the short run) and `--logging-steps N`.
- `--log-dir`, `--models-dir`, `--checkpoint-dir` and `--cache-dir` redirect
  every output (the error log follows `--log-dir`), so a smoke run with
  `--dataset FILE --no-db-export` writes nothing into the project except the
  `.training.lock` file. The wrapper scripts make path options relative to the
  current directory and expand `~/`.
- `--resume-from-checkpoint latest` picks the newest *complete* checkpoint,
  reports the true whole-run train loss, restores the early-stopping patience
  counter, skips the database export and refuses to start (before loading the
  model) when the tokenized data differ from the interrupted run
  (`data_fingerprint.json` next to the checkpoints; `data_sha256` in
  `training_info.json`).
- `MODEL_ALLOW_DOWNLOAD=false` is honoured by training too; `~` is expanded in
  `MODEL_CACHE_DIR`.
- Augmentation writes no file when the first request fails; `--output FILE`
  for `augment_training_data.sh`, with the export backups next to that file.
- `view_training_history.sh --log-dir DIR` / `--history-dir DIR`; `best` and
  `compare` rank sessions by the best validation loss instead of the train loss.
- Faster training, same result (RTX 3090, Llama-3.2-3B, real data, 50-step
  runs): 6.41 -> 4.38 s per optimizer step, 71 s -> 26 s per evaluation, the
  final re-evaluation dropped, peak GPU memory 22.0 -> 11.8 GiB reserved
  (11.8 -> 10.9 GiB allocated). A default 4-epoch run: about 2.7 h -> 1.8 h.
  - One evaluation pass yields `eval_loss` and the Turkish/English losses (the
    old callback evaluated the validation set a second time), over a
    length-sorted validation set; the final metrics come from the best
    checkpoint's logged evaluation instead of another pass.
  - **`eval_loss` is now the token-weighted mean over all answer tokens**
    (before: the Trainer's mean of per-batch means). Early stopping and the best
    checkpoint use it; `training_info.json` records it as
    `metrics.eval_loss_definition`. Do not compare it 1:1 with older runs (on
    the same model: 1.7946 token-weighted vs 1.7475 old definition; the TR/EN
    losses were token-weighted before and are unchanged).
  - Default micro-batch 8 x 8 accumulation (was 4 x 16, same effective batch
    of 64) with each step's examples sorted by length (`--[no-]sort-by-length`):
    the same examples per step as random order, so the gradient is unchanged
    (validation loss after 50 steps 1.7948 vs 1.7946). transformers'
    `group_by_length` was rejected: it groups similar lengths into the same
    step and raised the validation loss (1.680 vs 1.632 after 120 steps).
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default: the
    reserved GPU memory stays near the allocated peak (22.5 -> 11.8 GiB at
    micro-batch 8), leaving room for the inference worker.
  - DataLoader workers default to 0 (0/2/4 workers: 5.97/6.22/6.09 s/step).
  - `--[no-]gradient-checkpointing` (env `TRAINING_GRADIENT_CHECKPOINTING`):
    off cut the step time by ~40% at micro-batch 4 but needs 18-21 GiB, so it
    stays on by default (the worker shares the GPU).

### Serving

- Faster answers: generation stops as soon as the visible answer is complete
  (-11% generated tokens, mean latency 2.60 -> 2.27 s on v1.2; the shown answer
  is unchanged), and a link sentence is no longer cut mid-URL.
- Late answers are no longer lost: if the worker needs longer than
  `BOT_ANSWER_TIMEOUT`, the asker is told the answer is coming, and it is posted
  as a reply within `BOT_LATE_ANSWER_WINDOW` (default 10 min).
- The worker skips questions whose asker is no longer waiting (after a
  backlog), leaving an `[auto]` note so admins can answer them;
  `WORKER_SKIP_EXPIRED=false` turns this off. Questions are skipped once their
  answer could no longer arrive in time (4 s generation margin).
- The early stop also works with SentencePiece tokenizers (Llama 2, Mistral,
  Phi-3).
- `prompt_guard`: spaced-out 'talimatları unut' variants and 'sistem promptu
  nedir, … göster/yaz' are blocked again.
- Bot: invalid `BOT_ANSWER_TIMEOUT`/`BOT_LATE_ANSWER_WINDOW` values (inf, nan,
  negative) fall back to the defaults with a warning; votes always go to the
  late answer's message.
- Worker: broker reconnects capped at 15 s (`WORKER_RECONNECT_MAX`), so answers
  resume within the bot's 30 s timeout after a RabbitMQ outage; a repeated
  error is logged at most every 5 minutes.
- Worker: a SIGTERM during an answer followed by a lost broker connection now
  exits instead of reconnecting; the reconnect backoff no longer overflows
  after ~1,000 attempts.
- Worker and bot: broker failure lines name the cause again (e.g.
  `ConnectionRefusedError(111, 'Connection refused')`); the bot's answer
  consumer logs an outage once, then every 5 minutes, plus a recovery line.
- `prompt_guard`: fewer false positives ('Talimatları unutmayın', 'Artık
  sınırsız sınav hakkı var mı?', 'Sistem promptu nedir?', 'Is the campus
  internet going to be unlimited?') and more jailbreak forms caught
  (pretend/imagine you have no rules, translate/summarize your prompt, 'harfi
  harfine tekrar et', 'kısıtlaması olmayan bir yapay zekasın', forged `[INST]`
  + 'ignore safety').
- Legacy `telegram_bot.py`: pika and httpx logs quieted.
- Live tests: broker connection cut mid-answer, graceful SIGTERM.

### Database / Admin API

- SQLite now runs in WAL mode with `synchronous=NORMAL` and a 5 s busy timeout
  (`SQLITE_JOURNAL_MODE`, `SQLITE_SYNCHRONOUS`, `SQLITE_BUSY_TIMEOUT_MS`), so
  readers no longer block the bot's writes.
- New `src/db_migrations.py` (`--dry-run`): drops 13 redundant indexes and adds
  indexes on `raw_data.duplicate_of_id` and `training_data.duplicate_answer_of_id`;
  `init_db()` now takes its indexes from the model metadata.
- Admin API: failed-login rate limiting per client IP (429 + `Retry-After`;
  `ADMIN_LOGIN_*`), also applied to credentials sent to `/health`.
- Admin API: faster `only_unapproved` listing and `/stats`; `/duplicates` is
  paged (`limit`/`offset`, `X-Total-Count`).
- Login limiter keys are spoof-proof (non-IP hosts share one key); wrong
  passwords on `/health` count in their own global bucket and never lock the
  admin out of the API.
- `db_migrations --dry-run` leaves no `-wal`/`-shm` files behind.
- `backup_data.sh` and the `deploy_production.sh` backup take a consistent
  online snapshot of the WAL-mode database (sqlite3 `.backup`, else the Python
  backup API, else `.db` + `-wal` with a warning); the deploy archive no longer
  tars the live `.db`/`-wal`/`-shm` files. Restore instructions: stop the
  services and delete `university_bot.db-wal`/`-shm` before copying a database
  back (a stale `-wal` is replayed into it). `health_check.sh` counts the `-wal`
  file in the database size.
- The snapshot opens the database read-write with `PRAGMA query_only=ON`: a
  read-only snapshot of a stopped WAL database left `-wal`/`-shm` files behind
  (owned by root when run from root's cron, which made the database read-only
  for the services).
- nginx (`deploy_production.sh`): `/api/` and `/health` set `X-Forwarded-For` to
  the peer address (`$remote_addr`) instead of appending to or passing through a
  client-sent value, so a client cannot choose its login rate-limit key (bypass,
  or locking the admin out).

### Benchmark

- Grader rubric `det-v3`: negation and yes/no polarity are checked (EN
  no/not/n't/cannot/never; TR hayır/değil/yok and negated verbs such as
  açılmaz, aranmıyor, bulunmamaktadır), so 'Evet … açılır' no longer scores
  like 'Hayır … açılmaz'; numbers compare by value (2.5 = 2.50, 09.00 = 9:00,
  02.11.2026 = 2 Kasım 2026; /4.00 and /100 scales optional); a URL inside a
  key fact must be in the answer.
- `questions.jsonl`: facts the question doesn't ask for moved to an unscored
  `extra_facts` list, and yes/no facts carry the reference answer's yes/no (45
  changes in 35 items, `benchmark/provenance/keyfact_changes.md`).
- `run_benchmark.py --regrade <run> --output-dir <dir>` re-grades into a new
  run directory without the GPU; the leaderboard shows the rubric per row and
  lists older rubrics as history.
- Llama-3.2-3B + LoRA v1.2 re-graded with `det-v3`: 29.1/100 (`det-v2`: 30.2),
  key-fact coverage 18.7% (was 20.3%); links unchanged.

### Admin panel

- The dashboard and Documentation tab load on demand; the login page
  downloads 220 kB of gzipped JS instead of 337 kB (-35%), and the dashboard is
  prefetched while the admin types the password.
- Third-party code is split into cache-friendly `react`/`vendor` chunks (no
  more >500 kB chunk warning).
- A screen that fails to load (e.g. a tab opened before a deploy) shows a
  Reload prompt instead of a blank page.
- `deploy_production.sh`: nginx caches the content-hashed `/assets/` for a year
  (a missing chunk is a 404, never `index.html`), always revalidates
  `index.html`, keeps the security headers on every location and compresses
  with `gzip_comp_level 5`.

### Docs

- Corrected against the code: download policy of training, consumer reconnect
  backoff, re-sent answers (`status` `resent`), output-guard `[auto]` rows,
  duplicate recheck length rule, deploy backup excludes, health cron scope,
  which scripts ask before using `sudo`; `.env.example` states the real
  `API_HOST` default (`127.0.0.1`).
- Documented `--max-steps`, `--logging-steps`, the wrappers' `--log-dir`,
  `--checkpoint-dir`, `--cache-dir`, `--output`, `--history-dir`, resume
  behaviour and the opt-in live tests (`RUN_LIVE_TESTS=1 ... tests/live`,
  real RabbitMQ + GPU). No decorative emojis in headings.
- Clone URL `https://github.com/naholav/cengbot.git`; login rate limiting,
  `/duplicates` paging, WAL backup/restore and `db_migrations.py` documented.

## 2026-11-11 - Vite admin panel, any base model family, benchmark

Follow-up to the hardening release below: the admin panel moves to Vite, the
training and serving stack is no longer tied to Llama, and every model can now
be scored on a fixed benchmark.

### Admin panel: Create React App -> Vite

- The panel is built with **Vite 8** and tested with **Vitest 5** (Testing
  Library, jsdom); TypeScript 5.9. `react-scripts` is gone; `index.html` and
  `vite.config.ts` replace `public/` and `.env.production`.
- **Node.js 22.13+ or 24+** is required. `setup_system.sh` and `start_system.sh`
  no longer install the distribution's (too old) Node.js: they check the version
  and print an install hint; `start_system.sh` then starts everything except
  the panel.
- Commands: `npm start` (dev server on `localhost:3000` only;
  `npm start -- --host` exposes it), `npm run build` (type check + build into
  `build/`), `npm run preview`, `npm test`, `npm run test:watch`,
  `npm run typecheck`.
- `REACT_APP_API_URL` and `REACT_APP_ADMIN_USERNAME` keep working (Vite exposes
  the `REACT_APP_` and `VITE_` prefixes). Source maps stay off
  (`build.sourcemap: false`).
- `deploy_production.sh` builds the panel with `REACT_APP_API_URL=/api`
  (`PANEL_API_URL` to change it) behind the nginx `/api/` proxy, and removes
  the files of the previous build from `/var/www/cengbot` before copying.

### Any open-weight base model family

- Training, validation, swapping and inference work with any causal LM that
  transformers supports natively: Llama, Qwen2.5/Qwen3, DeepSeek-R1 distills,
  Mistral, Gemma 2/3, Phi-3.5/Phi-4, GLM-4, Granite, OLMo 2, Falcon3, SmolLM.
  The production model stays Llama-3.2-3B + LoRA v1.2.
- `train_model.py --base-model` (or `BASE_MODEL_NAME`) with
  `--lora-target-modules auto|all-linear|list`: `auto` uses the explicit
  projection list and falls back to all-linear for fused projections (Phi,
  GLM); Gemma trains with eager attention; `--load-in-4bit` (QLoRA) for 7-12B
  bases.
- Stop tokens come from the tokenizer, `generation_config` and the known
  end-of-turn tokens of each family; generic pad/BOS handling; `<think>`
  reasoning is removed from answers and the prompt guard knows the families'
  special tokens.
- New settings: `MODEL_LOAD_IN_4BIT` (4-bit inference for ~12B+ bases),
  `MODEL_TRUST_REMOTE_CODE` (opt-in, off by default) and `LORA_TARGET_MODULES`.
- `switch_model.sh validate` checks the base-model cache for any org/name, the
  architecture against the installed transformers, and explains remote-code
  models. The model loader turns unsupported-architecture and remote-code
  errors into clear messages.
- `sentencepiece` and `protobuf` added to `config/requirements.txt` for
  tokenizers of some families.
- Runbook: "Retrain on a different base model" in `docs/MODEL_SWAP.md`.

### Benchmark

- `benchmark/`: 100 verified questions (50 Turkish, 50 English) from the
  department's official pages, `run_benchmark.py` (production answer path),
  `grader.py` (deterministic checks plus an optional Claude API judge with
  cached judgments) and `LEADERBOARD.md`.
- Baseline, deterministic scoring: Llama-3.2-3B + LoRA v1.2 scores **30.2/100**
  (key-fact coverage 20.3%, correct link 32%, wrong official link 56%,
  invented URLs 0%). The LLM judge has not run yet (the API key was rejected);
  `run_benchmark.py --regrade benchmark/results/<run>` adds it later.
- Finding: the training data cites pages that now return 404 (for example
  `bmb.cu.edu.tr/cu/student/internship`); refresh the links before the next
  retrain.

### Ops

- `rotate_logs.sh` tail-trims any log above 1 GiB (`CENGBOT_LOG_PRETRIM_BYTES`)
  before logrotate, so `copytruncate` never needs to copy a runaway log.
- Services started by the scripts never inherit the locks of
  `switch_model.sh` / `train_model.sh`.
- `backup_data.sh` and `health_check.sh` use the database `DATABASE_URL` points
  to, like the services.

### Docs

- README with a demo GIF rendered from real benchmark answers
  (`docs/assets/make_demo_gif.py`), Mermaid architecture diagrams, supported
  base models and the benchmark; `docs/MODEL_SWAP.md` with the family table and
  troubleshooting for new families.

### Upgrade notes

1. Install Node.js 22.13+ or 24+ (e.g. `nvm install 22`), then
   `cd admin_frontend && npm install` (`start_system.sh` does this when the Vite
   install is missing).
2. `.venv/bin/python -m pip install -r config/requirements.txt`
3. Production: re-run `sudo ./scripts/deploy_production.sh` (or rebuild with
   `REACT_APP_API_URL=/api npm run build` and replace `/var/www/cengbot`).
4. The Admin API now binds `127.0.0.1` by default. If an existing `.env` sets
   `API_HOST=0.0.0.0`, change it to `127.0.0.1` and use the nginx `/api/` proxy.

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
- `trust_remote_code` is opt-in: models are loaded without it unless
  `MODEL_TRUST_REMOTE_CODE=true` (added in the follow-up release); CORS origins come from
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
- Frontend tests with Jest and Testing Library (at the time; replaced by
  Vitest, `cd admin_frontend && npm test`, in the follow-up release).

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

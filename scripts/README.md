# CengBot scripts

All scripts work from any directory (they locate the project from their own
path) and use the project interpreter `.venv/bin/python`. Shared helpers live
in `_common.sh` (sourced, not executed). Every script except
`health_check.sh` and `test_environment.sh` (which take no options and only
read) prints its options with `--help`.

Scripts that need root use plain interactive `sudo` (you are asked for **your**
password); no password is stored in any script. System-wide or destructive
scripts show what they will do and ask first (`--yes` skips the question,
`--dry-run` changes nothing). Without a terminal, a question counts as "no".

| Script | What it does | Changes the system? |
|---|---|---|
| `start_system.sh` | Start worker, Admin API, Telegram bot, Vite dev server of the admin panel (manual mode) | starts processes; `sudo` only to install/start RabbitMQ |
| `stop_system.sh` | Stop the processes of **this** project (verified PIDs, never other processes) | stops processes |
| `health_check.sh` | Services, `/health`, active model, database file (`DATABASE_URL`), disk/memory/GPU, log sizes, ports | read-only |
| `switch_model.sh` | List / validate / switch / roll back the active model, restart only the AI worker | `models/active-model` symlink |
| `train_model.sh` | Checks + `src/train_model.py` (any base model family) + validation of the new model directory | GPU training, `models/`, `checkpoints/` |
| `view_training_history.sh` | `list` / `view vN` / `compare vA vB` / `best` (by validation loss) for `logs/training_history/` (`--log-dir DIR`, `--history-dir DIR`) | read-only |
| `export_training_data.sh` | DB -> `data/cengbot_qa_augmented.jsonl` via `src/database_to_training.py` | `data/` |
| `augment_training_data.sh` | Claude API augmentation of the approved training data (costs money, asks first) | `data/` |
| `export_database.sh` | DB -> Excel (`excel/`) via `src/export_to_excel.py` | `excel/` |
| `backup_data.sh` | `backups/backup_<ts>.tar.gz` (DB snapshot, data, models, log tails) | `backups/` |
| `rotate_logs.sh` | Size-based log rotation without root (logrotate; tail-trims logs above 1 GiB first) | `logs/` |
| `test_environment.sh` | Python packages, CUDA, base model cache, datasets, active model | read-only |
| `cleanup_system.sh` | Python caches; optional temp files / logs / node_modules / build / model cache | asks per item |
| `setup_system.sh` | apt packages, Node.js version check, venv (only if missing), dependencies, DB tables, `.env` | system packages |
| `optimize_linux.sh` | sysctl, limits, GPU persistence, CPU governor, systemd units, logrotate | **system-wide**, root |
| `deploy_production.sh` | backup, systemd units, nginx, monitoring cron, optional firewall | **system-wide**, root |

## Running the system

```bash
./scripts/start_system.sh            # manual mode; logs are APPENDED to logs/*.log
./scripts/start_system.sh --no-frontend --install-deps
./scripts/health_check.sh
./scripts/stop_system.sh
```

`start_system.sh` installs dependencies only when the venv / `node_modules`
(Vite) is missing, or with `--install-deps`; `--no-frontend` skips the Vite dev
server of the admin panel (`npm start`, `localhost:3000` only, output in
`logs/frontend.log`). The panel needs **Node.js 22.13+ or 24+**: when Node.js
or npm is missing or too old, the script prints an install hint (nvm or
nodejs.org; the apt package is too old) and starts the other services without
the panel. `setup_system.sh` checks the version the same way and never
installs Node.js itself.
It stops older instances of this project's services (and the legacy/standalone
bots) first, and refuses to start when the systemd units (`cengbot-*.service`)
are active - use one mode, not both. In manual mode a crashed service is not
restarted; for production use the systemd units
(`sudo ./scripts/optimize_linux.sh --only services,logrotate`, then
`sudo systemctl enable --now cengbot-worker cengbot-bot cengbot-admin cengbot-monitor`),
which have `Restart=always`. `stop_system.sh` does not stop systemd-managed
services; it prints the `systemctl stop` command instead. Services started by
the scripts never inherit the lock files of `train_model.sh` or
`switch_model.sh` (a long-running worker would otherwise hold the lock).

## Switching the model (new training -> production)

See **[docs/MODEL_SWAP.md](../docs/MODEL_SWAP.md)** for the full runbook. Short version:

```bash
./scripts/switch_model.sh list                      # models + which one is active
./scripts/switch_model.sh validate final-best-model-v2
./scripts/switch_model.sh switch final-best-model-v2   # or: switch v2 / switch /path/outside/models
./scripts/switch_model.sh current
./scripts/switch_model.sh rollback                  # back to the previous model
./scripts/switch_model.sh restart                   # restart only the AI worker
```

`switch` validates the model directory, repoints `models/active-model`
atomically, remembers the previous model (`models/.previous-model`) and
restarts **only the AI worker** (manual mode: asks first, `--yes` to skip;
systemd mode: `sudo systemctl restart cengbot-worker`; `--no-restart` to only
print the command). The worker loads the model at start-up, so a
switch is not effective until the worker restarts. `--dry-run` shows the plan
without changing anything. Models are never deleted. Options: `-n/--dry-run`,
`-y/--yes`, `--no-restart`, `--name NAME`, `--move`.

## Training and data

```bash
./scripts/export_training_data.sh [--approved-only] [--exclude-disliked] [--replace] [--include-inactive] [--no-backup] [--output FILE]
./scripts/augment_training_data.sh [--dry-run] [--max-items N] [--model ID] [--reset] [--output FILE] [--yes]
./scripts/train_model.sh --dry-run        # data + tokenization checks, nothing is written
./scripts/train_model.sh                  # several hours on the GPU (asks; --yes to skip)
./scripts/train_model.sh --dataset data/merged_qa_data.jsonl --epochs 3   # other options go to train_model.py
./scripts/view_training_history.sh list | view v1 | compare v1 v2 | best
./scripts/view_training_history.sh --log-dir /tmp/exp/logs list    # a run trained with --log-dir /tmp/exp/logs
```

- `export_training_data.sh` passes every option to `src/database_to_training.py`.
  By default it merges into the existing file and keeps a backup in `backups/`
  next to it (`data/backups/` for the default file).
- `augment_training_data.sh` checks the database, shows a cost warning and asks
  you to type `YES`, runs `src/data_augmentation.py` and then exports the
  training data. The API key is read from `ANTHROPIC_API_KEY` / `CLAUDE_API_KEY`
  or asked for (hidden); it is never stored. A lock file prevents two runs at once.
  `--output FILE` writes the augmented data (and its resume checkpoint) to FILE
  instead of `data/cengbot_qa_augmented.jsonl`; the export afterwards merges into the same file.
- `train_model.sh` options: `-n/--dry-run`, `-y/--yes`; `--dataset FILE...`,
  `--models-dir DIR`, `--output-name NAME`, `--base-model ID|DIR`,
  `--checkpoint-dir DIR`, `--log-dir DIR`, `--cache-dir DIR` and
  `--resume-from-checkpoint latest|DIR` are checked by the wrapper (paths are
  relative to your current directory); every other option is passed to
  `src/train_model.py` (`.venv/bin/python src/train_model.py --help`), for
  example `--load-in-4bit` (QLoRA for 7-12B bases),
  `--lora-target-modules auto|all-linear|a,b,c`, `--max-steps N` (stop after N
  optimizer steps: smoke runs) and `--logging-steps N` (train loss every N steps, default 10),
  `--no-gradient-checkpointing` (faster, but 18 GiB or more: only with the worker stopped)
  and `--no-sort-by-length`.
  `--log-dir DIR` moves every log (wrapper log in `DIR/training/`, history in
  `DIR/training_history/`, error log); `--checkpoint-dir DIR` the checkpoints;
  `--cache-dir DIR` the base-model cache (default `MODEL_CACHE_DIR`, `model_cache/`).
  The wrapper warns when the base model (any Hugging Face org/name, default
  `BASE_MODEL_NAME`) is not in `model_cache/` yet, and when a gated base would
  need `HUGGING_FACE_TOKEN`. Other base families: [docs/MODEL_SWAP.md](../docs/MODEL_SWAP.md#retrain-on-a-different-base-model).
  Output: `models/final-best-model-v<N>/` (adapter + tokenizer +
  `training_info.json`), checkpoints in `checkpoints/<name>/`, wrapper log in
  `logs/training/auto_training_<ts>.log`. `--resume-from-checkpoint latest`
  continues an interrupted run (same `--output-name`/`--checkpoint-dir`) without
  re-exporting the database, and stops if the data changed. The script validates the new
  directory and prints the `switch_model.sh switch ...` command; it never
  activates a model by itself. A lock (`.training.lock`) prevents two runs at once.

## Exports, backups, cleanup

```bash
./scripts/export_database.sh [-v|-q] [--all-tables | --tables raw_data,user_votes] [--output-dir DIR | --output FILE]
./scripts/backup_data.sh [--no-models] [--keep N] [--dry-run]
./scripts/cleanup_system.sh [--temp-files] [--clear-logs] [--clean-node-modules] [--clean-build] [--clean-model-cache] [--dry-run] [--yes]
```

- `export_database.sh`: default the `raw_data` table to
  `excel/cengbot_database_export_<ts>.xlsx`; keeps the 10 newest default
  exports (`KEEP_EXPORTS=N`); log in `logs/excel_export.log`.
- `backup_data.sh`: SQLite online-backup snapshot of the database
  `DATABASE_URL` points to (default `university_bot.db`), `data/`, `models/` (the
  `active-model` symlink stays a symlink), small logs completely and the last
  10 MiB of big logs, `.env.example` (never `.env`). Keeps the 5 newest archives
  (`--keep N` or `KEEP_BACKUPS`). `backups/` contains user data: do not commit it.
- `cleanup_system.sh`: by default only removes Python caches in `src`,
  `config`, `scripts` and `tests`; every optional action asks first.
  `--clean-model-cache` means the base model must be downloaded again.

## Log rotation

`worker.log`, `bot.log`, `admin.log`, `frontend.log`, `monitor.log`,
`excel_export.log`, `deployment.log` and `optimization.log` are written by
shell redirection or systemd `StandardOutput=append:`, so Python cannot rotate
them. They are rotated by **logrotate** with `config/logrotate/cengbot.conf`:
`size 50M`, `rotate 5`, `copytruncate`, `compress` ->
`logs/<name>.log.1.gz ... .5.gz`. (`error.log` and `system_monitor.log` are
rotated by Python's `RotatingFileHandler` and are therefore not in that config.)

Choose one way to run it:

* **No root (per user)** - hourly entry in your own crontab:

  ```bash
  ./scripts/rotate_logs.sh --install-cron     # adds: 17 * * * * .../scripts/rotate_logs.sh --quiet
  ./scripts/rotate_logs.sh --dry-run          # logrotate -d: shows what would happen, changes nothing
  ./scripts/rotate_logs.sh                    # rotate now
  ./scripts/rotate_logs.sh --remove-cron
  ./scripts/rotate_logs.sh --print-config     # the rendered config
  ```

  It runs `logrotate -s logs/.logrotate.state logs/.logrotate.conf` (the config
  rendered with this project's path). Before that it tail-trims every service
  log bigger than 1 GiB (`CENGBOT_LOG_PRETRIM_BYTES`) to its last 10 MiB as
  `<name>.log.1.gz`: `copytruncate` copies the whole file first, so a runaway
  multi-GB log would need that much free disk. Without logrotate installed it
  falls back to the built-in size check below.

* **System-wide (root)** - `sudo ./scripts/optimize_linux.sh --only logrotate`
  installs `/etc/logrotate.d/cengbot` (with `su <user> <group>`) and
  `/etc/cron.hourly/cengbot-logrotate`. `deploy_production.sh` does this too.

As a safety net, `start_system.sh` (and the worker restart in `switch_model.sh`)
also rotates a log that is already bigger than 50 MiB before starting the
service, keeping only its last 10 MiB as `<name>.log.1.gz` and up to 5
generations (it never reads the whole file). Tunables: `CENGBOT_LOG_MAX_BYTES`,
`CENGBOT_LOG_KEEP_BYTES`, `CENGBOT_LOG_GENERATIONS`.

## System-wide scripts

```bash
sudo ./scripts/optimize_linux.sh --dry-run [--only kernel,limits,gpu,cpu,services,logrotate] [--yes]
sudo ./scripts/deploy_production.sh --dry-run [--with-firewall] [--tune-system] [--yes]
./scripts/setup_system.sh [--upgrade] [--yes]
```

See [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md). `optimize_linux.sh` writes the
systemd units `cengbot-worker`, `cengbot-bot`, `cengbot-admin`,
`cengbot-monitor` but does not start them.

## Environment variables understood by the scripts

| Variable | Meaning |
|---|---|
| `CENGBOT_ROOT` | project root (default: parent of `scripts/`; used by the tests) |
| `CENGBOT_PYTHON` | interpreter (default: `.venv/bin/python`) |
| `CENGBOT_BASE_PYTHON` | interpreter used to create `.venv` (`setup_system.sh`, `start_system.sh`) |
| `CENGBOT_NO_RESTART=1` | `switch_model.sh` never restarts the worker |
| `CENGBOT_SERVICE_USER` | user for systemd units / logrotate `su` (default: owner of the project dir) |
| `CENGBOT_LOG_PRETRIM_BYTES` | `rotate_logs.sh`: tail-trim logs bigger than this before logrotate (default 1 GiB) |
| `CENGBOT_LOG_MAX_BYTES`, `CENGBOT_LOG_KEEP_BYTES`, `CENGBOT_LOG_GENERATIONS` | start-time safety net: rotate above 50 MiB, keep the last 10 MiB, 5 generations |
| `PANEL_API_URL` | `deploy_production.sh`: API URL baked into the panel build (default `/api`; pass it as `sudo PANEL_API_URL=... ./scripts/deploy_production.sh`) |
| `DATABASE_URL` | the SQLite database that `backup_data.sh` and `health_check.sh` use (same rules as the services) |
| `ASSUME_YES=1` | same as `--yes` |
| `KEEP_EXPORTS`, `KEEP_BACKUPS` | how many Excel exports / backup archives to keep |
| `NO_COLOR=1` | no ANSI colours |

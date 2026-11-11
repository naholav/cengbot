# 🤖 CengBot - AI-Powered University Assistant

<div align="center">

![CengBot Logo](https://img.shields.io/badge/CengBot-AI%20Assistant-blue?style=for-the-badge&logo=telegram)

**LLaMA 3.2 3B + LoRA** fine-tuned Telegram assistant for the Çukurova University Computer Engineering Department, with a review-and-retrain loop.

🤖 **Try the Bot**: [@cu_ceng_v1_bot](https://t.me/cu_ceng_v1_bot)  
🤗 **HuggingFace**: [Naholav](https://huggingface.co/Naholav)  
📧 **Contact**: arda.mulayim@outlook.com  
📅 **Last updated**: November 2026 ([CHANGELOG](CHANGELOG.md))

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-queue-FF6600?style=flat&logo=rabbitmq&logoColor=white)](https://rabbitmq.com)
[![SQLite](https://img.shields.io/badge/SQLite-database-003B57?style=flat&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

## 🚀 Overview

CengBot answers students' questions about courses, programs and the department
in a Telegram group, in Turkish or English. The answers come from LLaMA 3.2 3B
with a LoRA adapter fine-tuned on department Q&A data.

Every question and answer is stored. Admins review them in a web panel, correct
answers and approve good pairs for training; the approved data is exported,
optionally augmented with the Claude API, and used to train the next model
version, which is then swapped in with one command. The project started with
about 600 English and 600 Turkish question-answer pairs, which augmentation
expanded to more than 22,000 training pairs.

## ✨ Features

- **Telegram bot** (`src/telegram_bot_rabbitmq.py`): answers in one configured
  group (optionally one forum topic) and ignores private chats and other groups;
  👍/👎 voting (one vote per user, at most two changes); 5 messages per minute
  per user; handles many questions at once; long answers are split to fit
  Telegram's limit.
- **Inference worker** (`src/ai_model_worker.py` + `src/llama_model_handler.py`):
  LLaMA 3.2 3B + LoRA, loaded offline-first from `model_cache/`, Turkish/English
  detection, short answers (4 sentences by default). It retries model loading and
  RabbitMQ with exponential backoff, and a question that keeps failing gets an
  apology after 3 attempts instead of being requeued forever.
- **Prompt guard** (`src/prompt_guard.py`): rule-based jailbreak and
  prompt-injection filter for Turkish and English. Blocked or over-long
  questions never reach the model, and the user gets a polite refusal.
- **Admin API + panel** (`src/admin_rest_api.py`, `admin_frontend/`): FastAPI
  with HTTP Basic auth on every data endpoint, and a React + Ant Design panel to
  browse, edit, approve and delete questions, view training data, statistics and
  duplicates, and read the docs and logs.
- **Duplicate detection**: similar questions (TF-IDF cosine ≥ 0.45 against the
  previous 100) and similar training answers (≥ 0.85) are marked automatically.
- **Training pipeline** (`src/train_model.py`): the training prompt is
  byte-identical to the inference prompt, loss is computed on the answer and EOS
  tokens only, and the train/validation split keeps a question and its augmented
  variants on the same side.
- **Model versioning**: models live in `models/final-best-model-v<N>/`; the
  `models/active-model` symlink selects the one in production, and
  `scripts/switch_model.sh` validates, switches and rolls back.
- **Operations**: start/stop/health scripts, systemd units, log rotation, a
  system monitor, Excel export and backups.

## 🏗️ Architecture

```
 Telegram group
      │  question                                          answer + 👍/👎
      ▼                                                          ▲
┌──────────────────────────┐  questions queue  ┌──────────────────────────┐
│ telegram_bot_rabbitmq.py │ ────────────────▶ │ ai_model_worker.py       │
│ (production bot)         │                   │  ├─ prompt_guard.py      │
│                          │ ◀──────────────── │  └─ llama_model_handler  │
└────────────┬─────────────┘   answers queue   └────────────┬─────────────┘
             │ stores question, votes            stores answer│
             ▼                                               ▼
       ┌─────────────────────────────────────────────────────────┐
       │ SQLite: university_bot.db  (src/database_models.py)     │
       └─────────────────────────────▲───────────────────────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │ admin_rest_api.py (FastAPI) │ ◀── admin_frontend (React)
                      └─────────────────────────────┘
```

1. The bot stores the question in `raw_data`, publishes it to the `questions`
   queue and waits up to `BOT_ANSWER_TIMEOUT` seconds (default 30) for the
   answer. An answer router hands each answer to the handler waiting for that
   question, so many questions can be in flight at once.
2. The worker runs the prompt guard, generates the answer, stores it in the
   database and publishes it to the `answers` queue.
3. The bot replies in the group with like/dislike buttons. If no answer arrives
   in time, it sends a fallback reply; the worker still stores the late answer.

Only `src/telegram_bot_rabbitmq.py` is a production bot. `src/telegram_bot.py`
is a legacy entrypoint that no script starts, and `src/telegram_bot_standalone.py`
is an optional development bot that loads the model in-process. Never run either
of them next to the production bot: they use the same token and Telegram rejects
the second poller with `409 Conflict`.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## ⚡ Quick Start

**Requirements**: Linux (the scripts target Ubuntu), Python 3.11, Node.js 18+,
RabbitMQ, and an NVIDIA GPU with CUDA for inference and training (CPU works but
is very slow). Access to the gated
[`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B)
base model is needed once, to download it into `model_cache/`.

```bash
# 1. Clone
git clone https://github.com/naholav/cu_ceng_bot.git
cd cu_ceng_bot

# 2. Python 3.11 virtual environment (all scripts use .venv/bin/python)
python3.11 -m venv .venv
.venv/bin/python -m pip install -r config/requirements.txt
.venv/bin/python -m pip install -r config/requirements-dev.txt   # tests

# 3. Frontend
(cd admin_frontend && npm install)

# 4. RabbitMQ
sudo apt install rabbitmq-server && sudo systemctl enable --now rabbitmq-server

# 5. Configuration
cp .env.example .env && chmod 600 .env
#    set at least TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_CHAT_ID and an admin password (see below)

# 6. Database tables (creates missing tables, keeps existing data)
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from database_models import init_db; init_db()"

# 7. Check, then start
./scripts/test_environment.sh
./scripts/start_system.sh
```

`scripts/setup_system.sh` automates steps 2-6 on Ubuntu (apt packages, `.venv`
if missing, dependencies, database, `.env` from the template).

**Model files are not in the repository** (`models/`, `model_cache/` and
`data/` are gitignored). Put a trained model directory into `models/` and
activate it with `./scripts/switch_model.sh switch <name>`, or train one (see
[Training](#-training-and-data)). The first start downloads the base model into
`model_cache/` if `HUGGING_FACE_TOKEN` is set and downloads are allowed.

Once running: admin panel <http://localhost:3000>, API docs
<http://localhost:8001/docs>, health `./scripts/health_check.sh`.

## 🔐 Configuration

All settings come from `.env`. **[`.env.example`](.env.example) is the full,
commented reference**; values already set in the process environment win over
`.env`. The most important ones:

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather (required) |
| `TELEGRAM_ALLOWED_CHAT_ID` | The one group the production bot answers (default: the department group) |
| `TELEGRAM_TOPIC_ID` | Optional: only answer in this forum topic |
| `BOT_ANSWER_TIMEOUT` | Seconds the bot waits for the worker (default 30) |
| `RABBITMQ_HOST` / `_PORT` / `_USER` / `_PASSWORD` | Broker connection (empty user = guest, localhost only) |
| `QUESTIONS_QUEUE` / `ANSWERS_QUEUE` | Queue names, must match for bot and worker |
| `MODEL_PATH` | Optional override; empty = `models/active-model` |
| `HUGGING_FACE_TOKEN` | Only to download the gated base model or to train |
| `MODEL_ALLOW_DOWNLOAD` / `HF_HUB_OFFLINE` | `false` / `1` forbids any model download |
| `PROMPT_GUARD_ENABLED` / `PROMPT_GUARD_MAX_CHARS` | Prompt guard on/off (default on) and length limit (default 1000) |
| `API_HOST` / `API_PORT` | Admin API bind address (template: `127.0.0.1:8001`; code default if unset: `0.0.0.0`) |
| `CORS_ORIGINS` | Exact origins the admin panel is served from |
| `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH` or `ADMIN_PASSWORD` | Admin login (see below) |
| `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` | Only for data augmentation |

`LORA_MODEL_PATH` is legacy and ignored.

### Admin login

There is **no default password**. Set `ADMIN_USERNAME` (default `admin`) and
one of:

- `ADMIN_PASSWORD_HASH`: the SHA-256 hex digest of the password (preferred), or
- `ADMIN_PASSWORD`: the plain password (used only when no valid hash is set).

If neither is set (or the value is a placeholder), every login is rejected with
401 and the API logs an ERROR at startup. Use an **ASCII-only** password: HTTP
Basic credentials are decoded as ASCII, so a password with `ç`, `ğ`, `ş` ... can
never log in. Generate the hash without the password ending up in your shell
history, then restart the Admin API:

```bash
python3 -c "import hashlib,getpass; print(hashlib.sha256(getpass.getpass().encode()).hexdigest())"
```

### Admin panel build settings

`REACT_APP_API_URL` (default `http://localhost:8001`) and
`REACT_APP_ADMIN_USERNAME` are baked into the frontend **at build time** by
`npm start` / `npm run build`. They are read from the shell or
`admin_frontend/.env.local` / `.env.production.local`, not from the root `.env`,
so rebuild after changing them. `admin_frontend/.env.production` sets
`GENERATE_SOURCEMAP=false`, so production builds ship no source maps. The origin
the panel is served from must be listed in `CORS_ORIGINS`.

## 🛠️ Running and Operations

```bash
./scripts/start_system.sh                 # worker, Admin API, bot, React dev server (manual mode)
./scripts/start_system.sh --no-frontend   # without the React dev server
./scripts/start_system.sh --install-deps  # (re)install Python + npm dependencies first
./scripts/stop_system.sh                  # stops only this project's processes
./scripts/health_check.sh                 # read-only: services, /health, active model, disk/GPU, log sizes
```

In manual mode the services' output is appended to `logs/worker.log`,
`logs/bot.log`, `logs/admin.log` and `logs/frontend.log`, and the PIDs are kept
in `.pids`. A crashed service is not restarted.

**Production (systemd).** `sudo ./scripts/optimize_linux.sh --only services,logrotate`
installs the units `cengbot-worker`, `cengbot-bot`, `cengbot-admin` and
`cengbot-monitor` (`Restart=always`, output appended to `logs/*.log`) without
starting them; then run
`sudo systemctl enable --now cengbot-worker cengbot-bot cengbot-admin cengbot-monitor`.
Use one mode only: `start_system.sh` refuses to run while the units are active.
`scripts/deploy_production.sh` does the full setup (units, nginx, health cron,
optional firewall); see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

**Logs and rotation**

- `logs/error.log` (errors with full tracebacks logged through
  `src/error_handler.py`: model loading, Admin API database errors, training)
  is size-rotated by Python: 10 MB × 5 files.
- The stdout logs (`worker.log`, `bot.log`, `admin.log`, `frontend.log`,
  `monitor.log`, ...) are rotated by **logrotate** with
  `config/logrotate/cengbot.conf` (50 MB, 5 compressed generations,
  `copytruncate`). Run it as your user with `./scripts/rotate_logs.sh`
  (`--install-cron` for an hourly crontab entry, `--dry-run` to preview), or
  system-wide with `sudo ./scripts/optimize_linux.sh --only logrotate`.
- Safety net: `start_system.sh` (and the worker restart in `switch_model.sh`)
  rotates any log above 50 MiB before starting a service, keeping its last
  10 MiB (`scripts/_common.sh`).
- `.venv/bin/python src/system_monitor.py --check-logs` reports log sizes and
  free disk space (exit code 1 on an alert; `--json` for JSON, `--once` for a
  full status report). Thresholds: `MONITOR_*` in `.env.example`.

## 🔄 Model Swap

`models/active-model` is a symlink and the single source of truth for the model
the worker loads. Switching restarts only the AI worker; the bot and the API
keep running.

```bash
./scripts/switch_model.sh list                       # models, type, size, training info; marks the active one
./scripts/switch_model.sh validate final-best-model-v2
./scripts/switch_model.sh switch final-best-model-v2 # or: switch v2 (validates, switches, restarts the worker)
./scripts/switch_model.sh current                    # active model + validation
./scripts/switch_model.sh rollback                   # back to the previous model
./scripts/switch_model.sh restart                    # restart only the worker
```

Every command accepts `--dry-run`; `--yes` skips the questions. While the
worker reloads (a few minutes), users get the bot's fallback reply; their
questions stay in RabbitMQ and are answered and stored once the model is loaded.

**Step-by-step runbook (training → validation → switch → verification →
rollback): [docs/MODEL_SWAP.md](docs/MODEL_SWAP.md).**

## 🎓 Training and Data

```bash
# Approved database rows -> data/cengbot_qa_augmented.jsonl (merges by default)
./scripts/export_training_data.sh [--approved-only] [--exclude-disliked] [--replace] [--include-inactive] [--no-backup] [--output FILE]

# Optional: 15 Claude-generated variations per approved pair (costs money, asks first)
./scripts/augment_training_data.sh --dry-run          # no API calls, no writes
./scripts/augment_training_data.sh [--max-items N] [--model ID] [--reset] [--yes]

# Train -> models/final-best-model-v<N>/ (never activates the model)
./scripts/train_model.sh --dry-run                    # data + tokenization stats only
./scripts/train_model.sh [train_model.py options]

# Training sessions (logs/training_history/)
./scripts/view_training_history.sh list | view v1 | compare v1 v2 | best
```

**`src/train_model.py` options** (all optional): `--dataset FILE...`,
`--base-model`, `--cache-dir`, `--models-dir`, `--output-name`,
`--checkpoint-dir`, `--log-dir`, `--epochs` (4), `--batch-size` (4),
`--eval-batch-size` (16), `--grad-accum` (16), `--lr` (2e-4), `--max-length`
(512), `--truncation drop|truncate`, `--val-ratio` (0.07), `--evals-per-epoch`
(5), `--early-stopping-patience` (3), `--seed` (42), `--lora-r` (16),
`--lora-alpha` (32), `--lora-dropout` (0.1), `--load-in-4bit`, `--cpu`,
`--workers` (4), `--semantic-dedup`, `--no-db-export`,
`--resume-from-checkpoint DIR|latest`, `--overwrite`, `--dry-run`.

- Default datasets: `data/merged_qa_data.jsonl` + `data/cengbot_qa_augmented.jsonl`
  (or `TRAINING_DATASET`). Before a real run the script refreshes the second
  file from the database (`--no-db-export` skips this).
- Exact duplicates are always removed; `--semantic-dedup` also removes near
  duplicates with `sentence-transformers` (same language, answer similarity
  ≥ 0.94 and question similarity ≥ 0.98).
- The base model is loaded offline-first from `model_cache/`; a download needs
  `HUGGING_FACE_TOKEN` and is impossible with `HF_HUB_OFFLINE=1`.
- Output: adapter, tokenizer and `training_info.json` in
  `models/final-best-model-v<N>/` (written atomically); Trainer checkpoints in
  `checkpoints/<name>/`.

**Data augmentation** (`src/data_augmentation.py`, options `--model`,
`--max-items`, `--dry-run`, `--yes`, `--output`, `--base-path`, `--reset`):
the default model is `claude-sonnet-5-5` (`--model` or `CLAUDE_MODEL` to
change). The API key comes from `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`, or is
asked for without echo; it is never stored or printed. The server-side refusal
fallback is on by default (`CLAUDE_REFUSAL_FALLBACK=off` disables it), and the
optional `AUGMENT_CLAUDE_EFFORT=low|medium|high|xhigh|max` lowers thinking cost.
Progress is checkpointed, so a run can be interrupted and resumed.

**Excel export**: `./scripts/export_database.sh` writes the `raw_data` table to
`excel/cengbot_database_export_<timestamp>.xlsx` and keeps the 10 newest
exports. Options for `src/export_to_excel.py` are passed through:
`--tables raw_data,user_votes`, `--all-tables`, `--output-dir DIR`,
`--output FILE [--overwrite]`.

## 🧪 Testing

```bash
.venv/bin/python -m pip install -r config/requirements-dev.txt
.venv/bin/python -m pytest                    # ~940 tests, offline
RUN_GPU_TESTS=1 .venv/bin/python -m pytest    # also the real-model GPU test (opt-in)
SHELLCHECK=/path/to/shellcheck .venv/bin/python -m pytest tests/test_shell_scripts.py
(cd admin_frontend && CI=true npx react-scripts test --watchAll=false)   # frontend tests
```

`tests/conftest.py` uses a temporary SQLite database, a temporary log directory
and dummy tokens, and fails the run if the real `university_bot.db` changes.
The tests never contact Telegram, RabbitMQ, Hugging Face or the Claude API. The
shellcheck test runs when `shellcheck` is on `PATH` or `SHELLCHECK` points to
it, and is skipped otherwise.

## 🛡️ Security

- Secrets live only in `.env` (gitignored, `chmod 600`). No password or token is
  hardcoded, and the scripts never pipe a password into `sudo`.
- Every Admin API data endpoint requires HTTP Basic auth. Public: `GET /`,
  `GET /health` (status only; system and GPU details need auth) and the OpenAPI
  docs (`/docs`, `/redoc`, `/openapi.json`).
- The panel's document viewer (`/docs/read`) serves only an allowlist (Markdown
  files, `logs/*.log`, `scripts/*.sh`, at most 1 MiB); `.env`, the database and
  model files are never served.
- The panel keeps credentials in memory and `sessionStorage` (cleared when the
  tab closes), never in `localStorage`.
- **Bind the API to `127.0.0.1`** (`API_HOST`) and expose it only through an
  HTTPS reverse proxy. HTTP Basic sends the password with every request, and
  there is **no brute-force throttling** on the login yet.
- Set `MODEL_ALLOW_DOWNLOAD=false` or `HF_HUB_OFFLINE=1` in production.
- `config/requirements.txt` pins audited minimum versions. Upgrading the ML
  stack (torch ≥ 2.10, transformers ≥ 4.57.6, accelerate ≥ 1.15) is recommended
  together with the next retrain, after testing on the GPU; see the comments in
  that file.

Checklist: [PRODUCTION_SECURITY_CHECKLIST.md](PRODUCTION_SECURITY_CHECKLIST.md).

## 📁 Project Structure

```
cu_ceng_bot/
├── src/
│   ├── telegram_bot_rabbitmq.py   # PRODUCTION Telegram bot (RabbitMQ)
│   ├── ai_model_worker.py         # RabbitMQ consumer: guard -> model -> answer
│   ├── prompt_guard.py            # jailbreak / prompt-injection guard (TR + EN)
│   ├── llama_model_handler.py     # model loading (offline-first) and generation
│   ├── admin_rest_api.py          # FastAPI admin API (port 8001)
│   ├── database_models.py         # SQLAlchemy models, duplicate detection, votes
│   ├── train_model.py             # LoRA fine-tuning
│   ├── database_to_training.py    # approved DB rows -> JSONL
│   ├── data_augmentation.py       # Claude API augmentation
│   ├── export_to_excel.py         # DB -> Excel
│   ├── system_monitor.py          # host / log / disk monitor
│   ├── error_handler.py           # central error logging (logs/error.log)
│   ├── telegram_bot.py            # legacy bot entrypoint (not started by any script)
│   └── telegram_bot_standalone.py # optional dev bot (model in-process, no RabbitMQ)
├── admin_frontend/                # React + TypeScript + Ant Design admin panel
│   ├── .env.production            # GENERATE_SOURCEMAP=false
│   ├── public/
│   └── src/                       # App.tsx, components/, contexts/, hooks/,
│                                  # services/api.ts, types/, utils/, *.test.ts(x)
├── config/
│   ├── env_loader.py              # loads .env, typed settings
│   ├── logrotate/cengbot.conf     # logrotate template
│   ├── requirements.txt           # Python dependencies
│   └── requirements-dev.txt       # test tools (pytest)
├── scripts/                       # start/stop/health, switch_model, train, export, ... (see scripts/README.md)
├── tests/                         # pytest suite (offline)
├── docs/                          # API, ARCHITECTURE, DEPLOYMENT, DEVELOPMENT, MODEL_SWAP
├── .env.example                   # configuration reference
├── pytest.ini
├── CHANGELOG.md
├── PRODUCTION_SECURITY_CHECKLIST.md
└── README.md

Created at runtime (gitignored): .env, .venv/, models/ (final-best-model-v*/,
active-model -> ...), model_cache/, checkpoints/, data/, excel/, logs/,
university_bot.db, .pids
```

## 🐛 Troubleshooting

| Problem | Check |
|---|---|
| Bot does not answer | `./scripts/health_check.sh`; the group's chat id must equal `TELEGRAM_ALLOWED_CHAT_ID`; `logs/bot.log` |
| `409 Conflict` in `bot.log` | Another poller uses the same token (a second bot instance, or the legacy/standalone bot) |
| Fallback reply for every question | Worker not running or still loading: `tail -f logs/worker.log` |
| Model fails to load | `./scripts/switch_model.sh current`; `nvidia-smi`; base model in `model_cache/` |
| RabbitMQ errors | `sudo systemctl status rabbitmq-server`; `sudo rabbitmqctl list_queues` |
| Admin login always fails | `ADMIN_PASSWORD_HASH` / `ADMIN_PASSWORD` set? ASCII only? API restarted after editing `.env`? |
| Panel shows network/CORS errors | Panel origin in `CORS_ORIGINS`; `REACT_APP_API_URL` at build time |

## 📚 Documentation

- [docs/MODEL_SWAP.md](docs/MODEL_SWAP.md): model swap runbook
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): components, data flow, database, logging
- [docs/API.md](docs/API.md): Admin API endpoints and authentication
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): production setup (systemd, nginx, log rotation, backups)
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): development setup, conventions, tests
- [scripts/README.md](scripts/README.md): every script and its options
- [PRODUCTION_SECURITY_CHECKLIST.md](PRODUCTION_SECURITY_CHECKLIST.md)
- [CHANGELOG.md](CHANGELOG.md)
- Interactive API docs: <http://localhost:8001/docs> when the API runs

## 📄 License

This project is licensed under the **Apache License 2.0**.

## 👨‍💻 Author

**naholav**
- **Email**: arda.mulayim@outlook.com
- **University**: Çukurova University Computer Engineering Department
- **GitHub**: [naholav](https://github.com/naholav)
- **HuggingFace**: [Naholav](https://huggingface.co/Naholav)
- **Bot**: [@cu_ceng_v1_bot](https://t.me/cu_ceng_v1_bot)

## 🙏 Acknowledgments

- **Meta AI** for the LLaMA 3.2 3B base model
- **Hugging Face** for the transformers and PEFT libraries
- **Çukurova University** for domain expertise and support
- **Open Source Community** for ML tools and frameworks

<div align="center">

**Built with ❤️ for Çukurova University Computer Engineering Department**

</div>

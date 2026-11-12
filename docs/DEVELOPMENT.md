# CengBot Development Guide

## Setup

Requirements: Python 3.11, **Node.js 22.13+ or 24+** and npm (Vite 8 and
Vitest 5 need it; Ubuntu's apt package is too old, use
[nvm](https://github.com/nvm-sh/nvm) or [nodejs.org](https://nodejs.org)),
RabbitMQ, and a CUDA GPU if you want to run the real model (the tests do not
need one).

```bash
git clone https://github.com/naholav/cengbot.git
cd cengbot

python3.11 -m venv .venv                                   # the scripts always use .venv/bin/python
.venv/bin/python -m pip install -r config/requirements.txt
.venv/bin/python -m pip install -r config/requirements-dev.txt   # pytest, pytest-asyncio, httpx

(cd admin_frontend && npm install)

cp .env.example .env && chmod 600 .env                     # fill in a test bot token, admin password, ...
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from database_models import init_db; init_db()"
```

`config/requirements-dev.txt` does not include `config/requirements.txt`, so
installing it never upgrades torch or transformers in an existing environment.

## Running the components

All at once: `./scripts/start_system.sh` (`--no-frontend` without the Vite
dev server; with a missing or too old Node.js it skips the panel and prints an
install hint), stop with `./scripts/stop_system.sh`. Or one by one, from the
project root:

```bash
.venv/bin/python src/ai_model_worker.py          # needs RabbitMQ + the model (GPU)
.venv/bin/python src/telegram_bot_rabbitmq.py    # needs RabbitMQ + TELEGRAM_BOT_TOKEN
.venv/bin/python src/admin_rest_api.py           # http://localhost:8001 (API_RELOAD=true for auto-reload)
(cd admin_frontend && npm start)                 # Vite dev server, http://localhost:3000 (localhost only)
```

Use a **separate test bot token** and test group for development: two pollers
on the same token make Telegram answer `409 Conflict`. The same applies to
`src/telegram_bot_standalone.py` (a development bot that loads the model
in-process, without RabbitMQ) and the legacy `src/telegram_bot.py`: never run
them next to the production bot.

For more detail, raise the log level (the entrypoints have no debug flag):

```bash
LOG_LEVEL=DEBUG .venv/bin/python src/ai_model_worker.py
LOG_LEVEL=DEBUG .venv/bin/python src/telegram_bot_rabbitmq.py
```

(The Admin API always logs at INFO.)

## Code layout and conventions

```
src/                 application code; modules import each other WITHOUT a "src." prefix
                     (from database_models import ...); every entrypoint puts src/ and the
                     project root on sys.path itself
config/env_loader.py .env loading and typed settings
scripts/             bash tools; all source scripts/_common.sh
tests/               pytest suite
admin_frontend/src/  React + TypeScript (Ant Design, TanStack Query, axios), built with Vite
benchmark/           benchmark questions, grader and runner (see benchmark/README.md)
```

- **Configuration** goes through `config/env_loader.py`. It loads `.env` once
  (or the file named by `CENGBOT_ENV_FILE`); variables already in the process
  environment win, empty values fall back to the default, relative paths
  resolve against the project root, and unedited `.env.example` placeholders
  count as unset. Document every new variable in `.env.example`.

  ```python
  from config.env_loader import get_config

  config = get_config()
  config.questions_queue                       # QUESTIONS_QUEUE, default "questions"
  config.model_path                            # MODEL_PATH, default <root>/models/active-model
  config.get_int("BOT_ANSWER_TIMEOUT", 30)     # get_str / get_int / get_float / get_bool / get_list / get_path
  ```

- **Database** access uses short-lived sessions from
  `database_models.SessionLocal` (always closed, rolled back on errors). Times
  are stored as naive UTC. `init_db()` only creates missing tables and indexes;
  there are no migrations, so a new column on an existing table needs a manual
  `ALTER TABLE` on existing databases.
- **Logging**: one concise line per event. Code that retries (worker, bot
  consumer) logs the first failure as an error and repeats more briefly, so a
  long outage cannot fill the disk. Unexpected errors with tracebacks go
  through `src/error_handler.py` into the rotating `logs/error.log`.
- **Scripts** source `scripts/_common.sh` for `PROJECT_ROOT`, `PYTHON`,
  `print_*`, `confirm`, `as_root` (interactive sudo, never a stored password),
  `service_pids` (only processes of this project) and log rotation. New
  scripts should accept `--help`, and `--dry-run` / `--yes` if they change the
  system.

## Common changes

**Admin API endpoint.** Register data endpoints on the `protected` router in
`src/admin_rest_api.py`; it applies `verify_credentials` to every route. Only
add routes to `app` directly when they must be public. Add tests to
`tests/test_admin_api.py` (they use FastAPI's `TestClient` and the test
credentials `admin` / `test-admin-password` from `tests/conftest.py`). Update
[API.md](API.md) and `admin_frontend/src/services/api.ts` + `types/api.ts`.

**Prompts.** The system prompts and turn markers
(`TRAINING_SYSTEM_PROMPT_TR/EN`, `TRAINING_TURN_MARKERS`) are defined once in
`src/llama_model_handler.py`; `src/train_model.py` imports them, and
`tests/test_training_pipeline.py` checks that the training prompt is
byte-identical to the inference prompt. The deployed adapter was trained with
these exact prompts, so changing them means retraining.

**Prompt guard rules.** `src/prompt_guard.py`, tests in
`tests/test_prompt_guard.py`. False positives are worse than misses here
(students ask real questions); add a test for both the attack and a similar
harmless question.

**Model directory format.** `src/llama_model_handler.py`
(`resolve_model_spec`), the validator in `scripts/switch_model.sh` and
`save_final_model` in `src/train_model.py` must agree on the contract in
[MODEL_SWAP.md](MODEL_SWAP.md).

**Frontend.** The panel is built with Vite (`admin_frontend/vite.config.ts`,
entry page `admin_frontend/index.html`):

```bash
cd admin_frontend
npm start               # dev server, http://localhost:3000 (localhost only; npm start -- --host to expose it)
npm run build           # tsc --noEmit + vite build -> build/ (no source maps)
npm run preview         # serve build/ on http://localhost:3000
npm test                # Vitest, once
npm run test:watch      # Vitest, watch mode
npm run typecheck       # tsc --noEmit
```

`REACT_APP_API_URL` (default `http://localhost:8001`) and
`REACT_APP_ADMIN_USERNAME` are read when the dev server starts or the build
runs, from the shell or `admin_frontend/.env.local` (also `.env.development.local`
/ `.env.production.local`); a shell variable wins. Read them through
`import.meta.env` (typed in `src/vite-env.d.ts`); `process.env` does not exist
in Vite. Only `REACT_APP_*` and `VITE_*` variables reach the bundle, so never
put secrets in them. The dev server origin `http://localhost:3000` is in the
default `CORS_ORIGINS`. Credentials are kept in memory and `sessionStorage`
only (`src/services/api.ts`); keep it that way.

The dashboard and the Documentation tab are lazy-loaded (`React.lazy` in
`src/App.tsx` and `src/components/Dashboard.tsx`; spinner and a Reload prompt
when a chunk fails to load in `src/components/LazyScreen.tsx`), and third-party
code is split into `react` and `vendor` chunks (`build.rolldownOptions` in
`vite.config.ts`). Put new heavy screens behind `lazy()` too, so the login page
stays small.

**Base model families.** Family-specific behaviour lives in
`src/llama_model_handler.py` (stop tokens in `STOP_TOKEN_CANDIDATES`, BOS and
appended-EOS handling, 4-bit loading, remote-code errors), `src/train_model.py`
(LoRA target modules, eager attention for Gemma) and
`src/prompt_guard.strip_reasoning` (`<think>` blocks). Add a family to
`tests/test_model_families.py` when you touch one of them; see
[MODEL_SWAP.md](MODEL_SWAP.md#retrain-on-a-different-base-model).

## Testing

```bash
.venv/bin/python -m pytest                                  # whole suite (1,800+ tests)
.venv/bin/python -m pytest tests/test_prompt_guard.py -q    # one file
.venv/bin/python -m pytest -k switch                        # by name
RUN_GPU_TESTS=1 .venv/bin/python -m pytest -m gpu           # opt-in: loads the real model on the GPU
RUN_LIVE_TESTS=1 .venv/bin/python -m pytest tests/live      # opt-in: real RabbitMQ + GPU worker + uvicorn
SHELLCHECK=/usr/bin/shellcheck .venv/bin/python -m pytest tests/test_shell_scripts.py

cd admin_frontend && npm test                               # frontend (Vitest + Testing Library, jsdom)
cd admin_frontend && npm run typecheck                      # TypeScript
```

The Python tests run offline. `tests/conftest.py` (do not bypass it) runs
before any project module is imported and:

- puts `src/` and the project root on `sys.path`;
- sets dummy tokens and keys, `HF_HUB_OFFLINE=1`, `ADMIN_USERNAME=admin` and
  `ADMIN_PASSWORD=test-admin-password`, so the real `.env` secrets are never used;
- points `LOG_DIR` and `ERROR_LOG_FILE` to a temporary directory, so the real
  `logs/` files are never written;
- points the ORM at a temporary SQLite database (fixture `db_session` gives a
  fresh schema) and fails the run if the real `university_bot.db` changes.

Mock pika, Telegram, torch/transformers and the Anthropic client; never load
the real model outside tests marked `gpu` (tiny randomly initialised models
built from a config are fine). The `gpu` tests are skipped unless
`RUN_GPU_TESTS=1`, the `live` tests in `tests/live/` unless `RUN_LIVE_TESTS=1`
(they need a running RabbitMQ, the GPU and the cached base model, and use
throw-away queues and a copy of the database); the shellcheck test is skipped
unless `shellcheck` is on `PATH` or `SHELLCHECK` points to it. Markers are
defined in `pytest.ini` (`gpu`, `slow`) and `tests/live/conftest.py` (`live`).

| Test file | Covers |
|---|---|
| `test_admin_api.py` | endpoints, auth, pagination, `/docs/read` allowlist, CORS |
| `test_ai_model_worker.py` | backoff, poison messages, answer publishing |
| `test_telegram_bot_rabbitmq.py` | answer router, rate limit, message splitting, votes |
| `test_telegram_bot_legacy.py` | the legacy and standalone bots |
| `test_llama_model_handler.py` | model resolution, offline-first loading, post-processing (+ the opt-in GPU test) |
| `test_model_families.py` | Qwen, DeepSeek, Mistral, Gemma, Phi, GLM, ... with tiny random models: train, validate, load, generate (+ an opt-in QLoRA GPU test) |
| `test_switch_model_families.py` | `switch_model.sh` / `train_model.sh` with non-Llama bases in a temporary project |
| `test_benchmark.py` | question set schema, grader, LLM judge (mocked), benchmark runner with a fake model |
| `test_prompt_guard.py` | blocked attacks and allowed questions (TR + EN) |
| `test_training_pipeline.py` | prompt alignment, label masking, group split, resume, export, augmentation, training wrappers |
| `test_training_e2e_gpu.py` | opt-in (`RUN_GPU_TESTS=1`): `train_model.sh` runs 3 real steps of the cached base model (outputs in a temp dir), validate, history viewer, the production handler answers TR + EN |
| `tests/live/test_integration_live.py` | opt-in (`RUN_LIVE_TESTS=1`): bot publisher/router, real worker and Admin API over a real RabbitMQ |
| `test_database_models.py` | schema, duplicates, votes and `like` sync |
| `test_shell_scripts.py` | the bash scripts in a temporary project (+ shellcheck) |
| `test_env_loader.py`, `test_error_handler.py`, `test_export_to_excel.py`, `test_system_monitor.py`, `test_smoke_infra.py` | the remaining modules |

## Debugging

```bash
# What would the worker load? (filesystem only, no weights)
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from llama_model_handler import resolve_model_spec; print(resolve_model_spec())"

# Database reachable?
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from database_models import SessionLocal; from sqlalchemy import text; s = SessionLocal(); print(s.execute(text('SELECT COUNT(*) FROM raw_data')).scalar()); s.close()"

# RabbitMQ reachable? Queue lengths?
.venv/bin/python -c "import pika; pika.BlockingConnection(pika.ConnectionParameters('localhost')).close(); print('RabbitMQ OK')"
sudo rabbitmqctl list_queues

# Training data and tokenization without a GPU run
./scripts/train_model.sh --dry-run

# Log sizes / free disk
.venv/bin/python src/system_monitor.py --check-logs
```

Loading the real model (`model_instance.load_model()`) needs the GPU memory
the running worker uses; stop the worker first.

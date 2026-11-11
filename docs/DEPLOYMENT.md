# 🚀 CengBot Deployment Guide

This guide covers a single Linux server (the scripts target Ubuntu 22.04+ with
`apt` and systemd). The scripts print their options with `--help`
(`health_check.sh` and `test_environment.sh` take no options and only read).
System-wide scripts show their plan and ask before changing anything
(`--dry-run` changes nothing, `--yes` skips the question). Details on every
script: [scripts/README.md](../scripts/README.md).

## 📋 Requirements

- **Python 3.11** (the pinned web stack needs it), **Node.js 18+** and npm
- **RabbitMQ** (`rabbitmq-server`)
- **NVIDIA GPU** with a working driver and CUDA for inference and training
  (CPU works but is very slow)
- Disk space for the base model cache (`model_cache/`), the adapters
  (`models/`), checkpoints, logs and backups
- A Hugging Face account with access to the gated
  [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B),
  once, to download the base model
- A Telegram bot token from [@BotFather](https://t.me/BotFather), and the bot
  added to the department group

## 🛠️ 1. Install

```bash
git clone https://github.com/naholav/cu_ceng_bot.git
cd cu_ceng_bot
./scripts/setup_system.sh            # [--upgrade] [--yes]
```

`setup_system.sh` installs the apt packages (Python, Node.js, RabbitMQ,
logrotate, build tools; `sudo` asks for *your* password), enables RabbitMQ and
its management plugin, creates `.venv` only if it does not exist yet, installs
`config/requirements.txt` and the npm packages, creates the database tables and
copies `.env.example` to `.env` (mode 600) if `.env` is missing. It runs
`apt upgrade` only with `--upgrade`, because a full upgrade can replace the
NVIDIA driver / CUDA stack.

Make sure `.venv` is a **Python 3.11** environment. To create it from another
interpreter, set `CENGBOT_BASE_PYTHON=/path/to/python3.11` before running the
script.

Manual alternative:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r config/requirements.txt
(cd admin_frontend && npm install)
sudo apt install rabbitmq-server && sudo systemctl enable --now rabbitmq-server
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from database_models import init_db; init_db()"
cp .env.example .env && chmod 600 .env
```

## ⚙️ 2. Configure

Edit `.env`. [`.env.example`](../.env.example) documents every variable.
For production:

```bash
TELEGRAM_BOT_TOKEN=...                 # from @BotFather
TELEGRAM_ALLOWED_CHAT_ID=-100...       # the one group the bot answers
RABBITMQ_USER=cengbot                  # optional; empty = guest (localhost only)
RABBITMQ_PASSWORD=...

API_HOST=127.0.0.1                     # never expose the API directly
API_PORT=8001
CORS_ORIGINS=https://admin.example.edu # exact origin(s) the panel is served from

ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=<64 hex chars>     # see below; there is NO default password

MODEL_ALLOW_DOWNLOAD=false             # after the base model is cached (see below)
LOG_LEVEL=INFO
```

Admin password hash (ASCII-only password; it is read without echo):

```bash
python3 -c "import hashlib,getpass; print(hashlib.sha256(getpass.getpass().encode()).hexdigest())"
```

**Model.** `models/` and `model_cache/` are not in the repository. Copy a
trained model directory to the server and activate it, or train one
([MODEL_SWAP.md](MODEL_SWAP.md)):

```bash
mkdir -p models
./scripts/switch_model.sh switch /path/to/final-best-model-v1.2 --no-restart   # validates and copies it into models/
./scripts/switch_model.sh current
```

The base model is downloaded into `model_cache/` the first time the worker
loads the model, if `HUGGING_FACE_TOKEN` is set and downloads are allowed. You
can also copy an existing Hugging Face cache directory
(`models--meta-llama--Llama-3.2-3B`) into `model_cache/`. When it is cached,
set `MODEL_ALLOW_DOWNLOAD=false` (or `HF_HUB_OFFLINE=1`) so the server never
downloads model files. `./scripts/test_environment.sh` shows whether the base
model is cached.

## ▶️ 3. Run

Use **one** of the two modes. `start_system.sh` refuses to start while the
systemd units are active.

### Manual mode

```bash
./scripts/start_system.sh                 # worker, Admin API, bot, React dev server
./scripts/start_system.sh --no-frontend   # without the dev server
./scripts/stop_system.sh
./scripts/health_check.sh
```

Output is appended to `logs/worker.log`, `logs/bot.log`, `logs/admin.log` and
`logs/frontend.log`; PIDs are kept in `.pids`. A crashed service is **not**
restarted, so use systemd in production.

### systemd units

```bash
sudo ./scripts/optimize_linux.sh --only services,logrotate --dry-run   # show what would be written
sudo ./scripts/optimize_linux.sh --only services,logrotate
sudo systemctl enable --now cengbot-worker cengbot-bot cengbot-admin cengbot-monitor
```

This writes `/etc/systemd/system/cengbot-{worker,bot,admin,monitor}.service`
(not started by the script) and the log rotation (see below). The units run
`.venv/bin/python src/<script>.py` as the owner of the project directory
(`CENGBOT_SERVICE_USER` to change), with `Restart=always`, and append their
output to `logs/worker.log`, `bot.log`, `admin.log` and `monitor.log`. The
worker unit allows 60 s to stop (GPU memory release). There is no unit for the
React dev server; serve a static build instead (next section).

```bash
sudo systemctl status cengbot-worker cengbot-bot cengbot-admin cengbot-monitor
sudo systemctl restart cengbot-admin      # e.g. after editing .env
tail -f logs/worker.log                   # service output (journalctl shows only start/stop events)
```

The other `optimize_linux.sh` steps (`kernel`, `limits`, `gpu`, `cpu`) tune
sysctl, file limits, GPU persistence mode and the CPU governor; run them only
if you want those system-wide changes (`--only kernel,limits,...`, or no
`--only` for all steps).

### Full deployment with nginx

`scripts/deploy_production.sh` sets up the whole server:

```bash
sudo ./scripts/deploy_production.sh --dry-run                 # show the plan
sudo ./scripts/deploy_production.sh [--with-firewall] [--tune-system] [--yes]
```

1. **backup**: archive of the project (code, `.env`, data, database, models;
   without logs, `.venv`, `model_cache`, `node_modules`, build) in
   `<parent dir>/cengbot_deploy_backups/` (mode 600)
2. **stop**: the `cengbot-*` units and processes started by `start_system.sh`
3. **packages**: `nginx`, `rabbitmq-server`, `logrotate`, `curl`
4. **app**: `pip install -r config/requirements.txt` into `.venv` (which must
   exist), database tables, `npm install && npm run build`, the build copied to
   `/var/www/cengbot`, `.env` set to mode 600
5. **units**: `optimize_linux.sh --only services,logrotate` (all steps with `--tune-system`)
6. **nginx**: `/etc/nginx/sites-available/cengbot` on port 80: the panel from
   `/var/www/cengbot`, `/api/` → `http://127.0.0.1:8001/`, `/health`
7. **monitoring**: `/usr/local/bin/cengbot-health.sh` every 5 minutes (cron);
   it starts stopped services and restarts `cengbot-admin` when `/health` fails;
   output in `/var/log/cengbot-health.log`
8. **firewall**: only with `--with-firewall`: `ufw` allows SSH, 80 and 443 and
   denies other incoming traffic
9. **start**: enables and starts the units and checks `/health`

> ⚠️ **Set the panel's API URL before deploying.** `REACT_APP_API_URL` is
> baked into the build and defaults to `http://localhost:8001`, which only works
> in a browser on the server itself. Behind this nginx configuration the panel
> should call the API through `/api` on the same origin:
>
> ```bash
> echo "REACT_APP_API_URL=/api" > admin_frontend/.env.production.local   # gitignored, read by npm run build
> ```
>
> The build runs as the service user in a login shell, so a variable exported
> in your own shell is not seen. Because panel and API then share one origin,
> CORS is not involved.

**HTTPS.** The nginx site listens on plain HTTP (port 80), and HTTP Basic sends
the admin password with every request. Add TLS before using the panel over a
network, for example with Let's Encrypt (not automated by the scripts):

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d admin.example.edu
```

## 📝 Logs

- `logs/error.log` is rotated by Python itself (`src/error_handler.py`, 10 MB × 5).
- The service output logs (`worker.log`, `bot.log`, `admin.log`,
  `frontend.log`, `monitor.log`, `excel_export.log`, `deployment.log`,
  `optimization.log`) are rotated by logrotate with
  `config/logrotate/cengbot.conf`: `size 50M`, `rotate 5`, `copytruncate`
  (no service restart, so the model is not reloaded), `compress`.

```bash
./scripts/rotate_logs.sh --dry-run          # logrotate -d: show what would happen
./scripts/rotate_logs.sh                    # rotate now, as the current user (no root)
./scripts/rotate_logs.sh --install-cron     # hourly entry in your crontab (--remove-cron to undo)
sudo ./scripts/optimize_linux.sh --only logrotate   # system-wide: /etc/logrotate.d/cengbot + /etc/cron.hourly
```

Do not add `logs/*.log` to another logrotate rule: it would also rotate
`error.log` and `system_monitor.log`, which Python already rotates. As a safety
net, `start_system.sh` (and the worker restart in `switch_model.sh`) rotates a
log above 50 MiB before starting a service.

Check sizes and free disk space at any time:

```bash
.venv/bin/python src/system_monitor.py --check-logs     # exit code 1 if a threshold is exceeded
```

## 💾 Backups

```bash
./scripts/backup_data.sh                 # backups/backup_<timestamp>.tar.gz
./scripts/backup_data.sh --no-models --keep 10 --dry-run
```

The archive holds a consistent snapshot of the SQLite database (online backup
API), `data/`, `models/` (`--no-models` to skip), the last part of big logs and
`.env.example` (never `.env`). The 5 newest archives are kept (`--keep N` or
`KEEP_BACKUPS`). Copy the archives to another machine; `backups/` contains user
data, so never commit it.

## 🔄 Updates

```bash
./scripts/backup_data.sh
git pull
.venv/bin/python -m pip install -r config/requirements.txt
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from database_models import init_db; init_db()"   # adds missing tables only
sudo systemctl restart cengbot-worker cengbot-bot cengbot-admin cengbot-monitor
# frontend changes: rebuild and copy (or re-run deploy_production.sh)
(cd admin_frontend && npm install && npm run build) && sudo cp -a admin_frontend/build/. /var/www/cengbot/
```

In manual mode use `./scripts/stop_system.sh` and `./scripts/start_system.sh
--install-deps` instead of the `systemctl` and `pip` steps. Restarting the
worker reloads the model (a few minutes); questions asked meanwhile get the
bot's fallback reply.

The ML stack upgrade (torch ≥ 2.10, transformers ≥ 4.57.6, accelerate ≥ 1.15)
recommended in the comments of `config/requirements.txt` should be done
together with the next retrain, after testing on the GPU.

## 🔒 Hardening

- `.env` mode 600, owned by the service user (`deploy_production.sh` sets this).
- `API_HOST=127.0.0.1`; publish the API only through the nginx proxy with HTTPS.
  The API has no brute-force throttling yet; restrict the panel to trusted
  networks if you can.
- `MODEL_ALLOW_DOWNLOAD=false` or `HF_HUB_OFFLINE=1`.
- Use RabbitMQ credentials other than `guest` if the broker is not local.
- Keep `university_bot.db`, `logs/`, `backups/` and the deploy backups readable
  only by the service user.
- See [PRODUCTION_SECURITY_CHECKLIST.md](../PRODUCTION_SECURITY_CHECKLIST.md).

## 🐛 Troubleshooting

| Problem | What to check |
|---|---|
| A unit keeps restarting | `tail -n 100 logs/<service>.log`; `systemctl status <unit>` |
| Worker logs `Model load failed ... Retrying` | `./scripts/switch_model.sh current`; base model cached? `nvidia-smi`; GPU memory used by another process? |
| Bot logs `409 Conflict` / "Telegram connectivity problem: Conflict" | Another poller uses the token: a second instance, the legacy or the standalone bot (`./scripts/health_check.sh` warns) |
| Users always get the fallback reply | Worker not running or still loading; `sudo rabbitmqctl list_queues` (growing `questions` queue) |
| Panel: login fails | Password set and ASCII? API restarted after editing `.env`? |
| Panel: network errors | `REACT_APP_API_URL` used at build time; `CORS_ORIGINS` contains the panel origin (when the API is on another origin) |
| `start_system.sh`: `... is active (systemd mode)` | The units are running; use `systemctl` instead of the script |
| Database | `.venv/bin/python -c "import sqlite3; print(sqlite3.connect('university_bot.db').execute('PRAGMA integrity_check').fetchone())"` |

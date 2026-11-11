# 🔒 Production Security Checklist

Go through this list before the bot and the admin panel serve real users.
Background: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), [docs/API.md](docs/API.md).

## ✅ Before going live

### 1. Secrets and `.env`
- [ ] `.env` created from `.env.example`, mode 600, owned by the service user
- [ ] `TELEGRAM_BOT_TOKEN` set (a token that was ever committed or shared must be revoked with @BotFather)
- [ ] `TELEGRAM_ALLOWED_CHAT_ID` is the department group (the bot answers no other chat)
- [ ] `ADMIN_PASSWORD_HASH` (SHA-256 hex digest, preferred) or `ADMIN_PASSWORD` set,
      with a strong **ASCII-only** password. There is **no default password**:
      until one is set every admin login is rejected (401)
- [ ] `ANTHROPIC_API_KEY` / `CLAUDE_API_KEY` only on the machine that runs augmentation (or leave empty: the script asks)
- [ ] `HUGGING_FACE_TOKEN` empty once the base model is in `model_cache/`
- [ ] `LORA_MODEL_PATH` removed (legacy and ignored) and `MODEL_PATH` empty, so `models/active-model` decides

```bash
# Password hash (read without echo, never in the shell history)
python3 -c "import hashlib,getpass; print(hashlib.sha256(getpass.getpass().encode()).hexdigest())"
```

### 2. Repository hygiene
- [ ] `git status --short --ignored`: `.env`, `university_bot.db`, `logs/`,
      `models/`, `model_cache/`, `checkpoints/`, `data/`, `excel/` are ignored
      and not tracked (untrack with `git rm --cached` if needed)
- [ ] `backups/` and deploy backup archives are not committed (they contain user data and, for deploy backups, `.env`)
- [ ] No tokens, passwords or password hashes in docs, issues or commit messages

### 3. Network
- [ ] `API_HOST` is `127.0.0.1` (the default) - never `0.0.0.0` on a public host
- [ ] The panel and the API are reachable only through nginx **with HTTPS**
      (HTTP Basic sends the password with every request)
- [ ] `CORS_ORIGINS` lists only the real panel origin(s)
- [ ] The panel is built with the right `REACT_APP_API_URL` (`deploy_production.sh`
      uses `/api` behind its nginx site; `PANEL_API_URL` overrides it)
- [ ] Firewall: only SSH, 80 and 443 (`deploy_production.sh --with-firewall`)
- [ ] RabbitMQ: not reachable from outside; own credentials (`RABBITMQ_USER` /
      `RABBITMQ_PASSWORD`) if it is not on localhost. The management UI
      (port 15672, enabled by `setup_system.sh`) is not exposed

### 4. Model and dependencies
- [ ] `MODEL_ALLOW_DOWNLOAD=false` or `HF_HUB_OFFLINE=1`
- [ ] `MODEL_TRUST_REMOTE_CODE` unset or `false`. `true` executes Python code
      from the model repository; use it only for a legacy remote-code model
      from a repository you trust
- [ ] `./scripts/switch_model.sh current` reports a valid model
- [ ] `PROMPT_GUARD_ENABLED` not disabled
- [ ] Dependencies installed from `config/requirements.txt`; audit regularly:
      `pip install pip-audit && pip-audit -r config/requirements.txt` and
      `cd admin_frontend && npm audit --omit=dev`
- [ ] Plan the ML stack upgrade (torch ≥ 2.10, transformers ≥ 4.57.6,
      accelerate ≥ 1.15) together with the next retrain, after GPU testing
      (see the comments in `config/requirements.txt`). Load only model files you
      trust.

### 5. Operations
- [ ] systemd units installed and enabled (`Restart=always`)
- [ ] Log rotation installed (`./scripts/rotate_logs.sh --install-cron` or
      `sudo ./scripts/optimize_linux.sh --only logrotate`)
- [ ] Backups scheduled (`./scripts/backup_data.sh`) and copied off the server
- [ ] `./scripts/health_check.sh` and `./scripts/test_environment.sh` pass
- [ ] Test suite passes: `.venv/bin/python -m pytest`

## 🛡️ What the code already does

**Admin API and panel**
- HTTP Basic auth on every data endpoint; public only `GET /`, a minimal
  `GET /health` and the OpenAPI docs
- No default or hardcoded password; empty, placeholder or malformed values are
  treated as unset; constant-time comparison
- The document viewer serves only Markdown files, `logs/*.log` and
  `scripts/*.sh` (at most 1 MiB); never `.env`, the database or model files;
  no path traversal
- Database errors return a generic message (details only in the server log)
- The panel keeps credentials in memory and `sessionStorage` (cleared when the
  tab closes), never in `localStorage`, and logs out on 401; production builds
  ship without source maps (`build.sourcemap: false` in
  `admin_frontend/vite.config.ts`), and only `REACT_APP_*` / `VITE_*` variables
  reach the bundle

**Telegram bot**
- Answers only in `TELEGRAM_ALLOWED_CHAT_ID`; private chats and other groups are
  ignored (vote button presses from other chats are rejected and logged)
- Rate limit: 5 messages per user per minute
- User and model text is HTML-escaped; vote callbacks are validated and must
  come from the bot's own reply message

**Model**
- Prompt guard (Turkish + English) blocks jailbreak / prompt-injection attempts
  and over-long questions before they reach the model, and stops answers that
  leak the system prompt
- Offline-first model loading; downloads can be disabled completely
- Code from a model repository is never executed unless
  `MODEL_TRUST_REMOTE_CODE=true`

**Scripts**
- No stored passwords; root actions use interactive `sudo`
- System-wide changes are shown and confirmed first (`--dry-run`, `--yes`)
- Only processes of this project are stopped (PIDs verified against the command line)

## ⚠️ Known limitations

- **No brute-force throttling** on admin logins and no rate limiting in the
  API. Keep it behind the proxy; add rate limiting there if needed.
- The password hash is an unsalted SHA-256: use a long, random password.
- HTTP Basic has no logout on the server side; closing the tab clears the
  panel's copy of the credentials.

## 🔐 Emergency procedures

### Change the admin password
```bash
NEW_HASH=$(python3 -c "import hashlib,getpass; print(hashlib.sha256(getpass.getpass().encode()).hexdigest())")
sed -i "s/^ADMIN_PASSWORD_HASH=.*/ADMIN_PASSWORD_HASH=$NEW_HASH/" .env
unset NEW_HASH

# The API reads .env at start-up: restart it
sudo systemctl restart cengbot-admin                          # systemd mode
# manual mode: ./scripts/stop_system.sh && ./scripts/start_system.sh  (also restarts the worker)
```

### Security incident
1. **Contain**: stop the services (`sudo systemctl stop cengbot-worker cengbot-bot cengbot-admin cengbot-monitor`
   or `./scripts/stop_system.sh`).
2. **Assess**: check `logs/admin.log`, `logs/bot.log`, `logs/error.log` and the nginx logs.
3. **Rotate secrets**: Telegram token (@BotFather), admin password, RabbitMQ
   credentials, Hugging Face and Anthropic keys.
4. **Verify**: go through this checklist again; run `./scripts/health_check.sh`.
5. **Monitor**: watch the logs closely for the next days.

---

**Last updated**: November 2026

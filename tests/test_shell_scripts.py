"""
Tests for the shell scripts in scripts/.

* static checks for every script (bash -n, no CRLF, no embedded sudo password,
  no references to missing scripts, no broad pkill patterns, shellcheck)
* functional tests that run the scripts against a FAKE project directory
  (CENGBOT_ROOT=<tmp>), never against the real project: switch_model.sh
  (validate / switch / import / rollback / dry-run / worker restart),
  stop_system.sh (only this project's processes), log rotation (_common.sh and
  the logrotate config), backup / cleanup / train / export wrappers.

Nothing here touches the real models/, logs/, data/ or database, starts the
real services, uses sudo, or talks to the network.
"""

import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
ALL_SCRIPTS = sorted(SCRIPTS.glob("*.sh"))
SWITCH = SCRIPTS / "switch_model.sh"
LOGROTATE_TEMPLATE = ROOT / "config" / "logrotate" / "cengbot.conf"
TEXT_FILES = ALL_SCRIPTS + [SCRIPTS / "README.md", ROOT / "docs" / "MODEL_SWAP.md", LOGROTATE_TEMPLATE]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def script_env(root, **extra):
    env = dict(os.environ)
    env.update(
        CENGBOT_ROOT=str(root),
        CENGBOT_PYTHON=sys.executable,
        CENGBOT_NO_RESTART="1",
        NO_COLOR="1",
    )
    env.pop("MODEL_PATH", None)
    env.pop("ASSUME_YES", None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run(script, *args, root, cwd=None, timeout=120, **env):
    return subprocess.run(
        ["bash", str(script), *map(str, args)],
        cwd=str(cwd or root),
        env=script_env(root, **env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def out(proc):
    return proc.stdout + proc.stderr


def write_safetensors(path, truncated=False):
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    data = struct.pack("<Q", len(header)) + header + (b"" if truncated else b"\0\0\0\0")
    Path(path).write_bytes(data)


def make_lora(d, base="meta-llama/Llama-3.2-3B", info=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": base}))
    write_safetensors(d / "adapter_model.safetensors")
    if info:
        (d / "training_info.json").write_text(json.dumps(info))
    return d


def snapshot(root):
    """Everything about a tree that a dry run must not change."""
    state = {}
    for p in sorted(Path(root).rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            state[rel] = ("link", os.readlink(p))
        elif p.is_file():
            state[rel] = ("file", hashlib.sha256(p.read_bytes()).hexdigest())
        else:
            state[rel] = ("dir",)
    return state


@pytest.fixture
def fake_project(tmp_path):
    root = tmp_path / "proj"
    models = root / "models"
    make_lora(models / "final-best-model-v1" / "method1")
    (models / "final-best-model-v1" / "training_info.json").write_text(json.dumps({"final_loss": 0.8259}))
    make_lora(models / "final-best-model-v1.2", info={"final_loss": 0.7, "timestamp": "2025-07-31T10:00:00"})
    make_lora(models / "final-best-model-v2")
    broken = models / "broken"
    broken.mkdir()
    (broken / "adapter_config.json").write_text("{}")
    trunc = models / "truncated"
    trunc.mkdir()
    (trunc / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    write_safetensors(trunc / "adapter_model.safetensors", truncated=True)
    binonly = models / "binonly"
    binonly.mkdir()
    (binonly / "adapter_config.json").write_text("{}")
    (binonly / "adapter_model.bin").write_bytes(b"x")
    full = models / "full-merged"
    full.mkdir()
    (full / "config.json").write_text(json.dumps({"model_type": "llama"}))
    write_safetensors(full / "model.safetensors")
    (full / "tokenizer.json").write_text("{}")
    os.symlink("final-best-model-v1.2", models / "active-model")
    make_lora(tmp_path / "ext" / "new-model")
    (root / "logs").mkdir()
    (root / "src").mkdir()
    return root


def active(root):
    return os.readlink(Path(root) / "models" / "active-model")


# --------------------------------------------------------------------------
# static checks
# --------------------------------------------------------------------------
@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script):
    proc = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("path", TEXT_FILES, ids=lambda p: p.name)
def test_no_crlf_line_endings(path):
    assert b"\r" not in path.read_bytes(), f"{path} has CRLF line endings (bash cannot run it)"


def test_scripts_are_executable():
    for script in ALL_SCRIPTS:
        assert os.access(script, os.X_OK), f"{script.name} is not executable"


def test_no_embedded_sudo_password():
    for path in TEXT_FILES:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"sudo\s+(-\w*\s+)*-S\b", text), f"'sudo -S' (password on stdin) in {path}"
        assert not re.search(r"\|\s*sudo\b", text), f"something piped into sudo in {path}"


def test_no_hardcoded_wrong_paths_or_interpreters():
    for script in ALL_SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert not re.search(r"/home/ceng/cu_ceng_bot([\"'/]|$)", text, re.M), f"non-existent project path in {script.name}"
        assert "miniconda3/bin/python" not in text, f"wrong interpreter in {script.name}"
        assert "miniconda3/bin/pip" not in text, f"wrong pip in {script.name}"
        assert not re.search(r"^\s*python3? +src/", text, re.M), f"bare python (not .venv) in {script.name}"
        assert "from src.database_models" not in text, f"src.database_models import in {script.name}"


def test_no_broad_process_kill_patterns():
    for script in ALL_SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert "pkill" not in text and "killall" not in text, f"{script.name} kills processes by name"


def test_referenced_scripts_exist():
    pattern = re.compile(r"(?:scripts/|\$SCRIPTS_DIR/|\$SCRIPT_DIR/|\./)([A-Za-z0-9_]+\.sh)\b")
    for path in TEXT_FILES:
        for name in pattern.findall(path.read_text(encoding="utf-8")):
            assert (SCRIPTS / name).is_file(), f"{path.name} references missing scripts/{name}"
    assert not (SCRIPTS / "dev_mode.sh").exists()


def test_shellcheck_clean():
    shellcheck = os.environ.get("SHELLCHECK") or shutil.which("shellcheck")
    if not shellcheck:
        pytest.skip("shellcheck not installed (set SHELLCHECK=/path/to/shellcheck to run it)")
    proc = subprocess.run(
        [shellcheck, "-x", "-P", str(SCRIPTS), "-e", "SC1091", "-f", "gcc", *map(str, ALL_SCRIPTS)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_python_cli_flags_used_by_scripts_exist():
    """Flags the wrappers pass must exist in the Python modules' argparse."""
    train = (ROOT / "src" / "train_model.py").read_text(encoding="utf-8")
    for flag in ("--dry-run", "--dataset", "--models-dir", "--output-name"):
        assert re.search(r"add_argument\(\s*['\"]%s['\"]" % re.escape(flag), train), flag


# --------------------------------------------------------------------------
# switch_model.sh
# --------------------------------------------------------------------------
def test_list_marks_active_model(fake_project):
    proc = run(SWITCH, "list", root=fake_project)
    assert proc.returncode == 0, out(proc)
    line = next(l for l in proc.stdout.splitlines() if "final-best-model-v1.2" in l and "(ACTIVE)" in l)
    assert line.strip().startswith("*")
    assert "loss 0.7000" in line
    assert re.search(r"final-best-model-v1\s+LoRA adapter \(method1/\)", proc.stdout)
    assert re.search(r"full-merged\s+full model", proc.stdout)
    assert re.search(r"broken\s+INVALID", proc.stdout)


@pytest.mark.parametrize("name", ["final-best-model-v1", "final-best-model-v1.2", "full-merged", "v2", "1.2"])
def test_validate_accepts_contract_layouts(fake_project, name):
    proc = run(SWITCH, "validate", name, root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert "valid" in proc.stdout


@pytest.mark.parametrize("name,message", [
    ("broken", "adapter_model.safetensors is missing"),
    ("truncated", "truncated"),
    ("binonly", "adapter_model.bin"),
])
def test_validate_rejects_invalid_models(fake_project, name, message):
    proc = run(SWITCH, "validate", name, root=fake_project)
    assert proc.returncode == 1
    assert "NOT a valid model directory" in out(proc)
    assert message in out(proc)


def test_validate_warns_when_base_model_not_cached(fake_project):
    proc = run(SWITCH, "validate", "v2", root=fake_project)
    assert "not cached in model_cache" in proc.stdout
    cache = fake_project / "model_cache" / "models--meta-llama--Llama-3.2-3B" / "snapshots" / "abc"
    cache.mkdir(parents=True)
    (cache / "config.json").write_text("{}")
    proc = run(SWITCH, "validate", "v2", root=fake_project)
    assert proc.returncode == 0 and "not cached" not in proc.stdout


def test_switch_repoints_relative_symlink_and_records_previous(fake_project):
    proc = run(SWITCH, "switch", "final-best-model-v2", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert active(fake_project) == "final-best-model-v2"          # relative target
    models = fake_project / "models"
    assert (models / ".previous-model").read_text().strip() == "final-best-model-v1.2"
    assert "final-best-model-v1.2 -> final-best-model-v2" in (models / ".switch-history.log").read_text()
    assert "NOT restarted" in proc.stdout                          # CENGBOT_NO_RESTART=1
    assert not list(models.glob(".active-model.tmp*"))


def test_switch_version_shortcuts(fake_project):
    assert run(SWITCH, "switch", "v2", root=fake_project).returncode == 0
    assert active(fake_project) == "final-best-model-v2"
    assert run(SWITCH, "switch", "1.2", root=fake_project).returncode == 0
    assert active(fake_project) == "final-best-model-v1.2"


def test_switch_method1_path_points_to_model_dir(fake_project):
    proc = run(SWITCH, "switch", "models/final-best-model-v1/method1", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert active(fake_project) == "final-best-model-v1"


def test_switch_to_active_model_is_a_noop(fake_project):
    proc = run(SWITCH, "switch", "final-best-model-v1.2", root=fake_project)
    assert proc.returncode == 0 and "already the active model" in proc.stdout
    assert not (fake_project / "models" / ".previous-model").exists()


@pytest.mark.parametrize("target", ["broken", "truncated", "does-not-exist", "active-model"])
def test_switch_rejects_invalid_target_and_changes_nothing(fake_project, target):
    before = snapshot(fake_project)
    proc = run(SWITCH, "switch", target, root=fake_project)
    assert proc.returncode != 0
    assert snapshot(fake_project) == before
    assert active(fake_project) == "final-best-model-v1.2"


@pytest.mark.parametrize("args", [
    ["switch", "v2", "--dry-run"],
    ["--dry-run", "switch", "../ext/new-model"],
    ["-n", "switch", "../ext/new-model", "--yes", "--move"],
])
def test_dry_run_changes_nothing(fake_project, args):
    before = snapshot(fake_project.parent)
    proc = run(SWITCH, *args, root=fake_project, cwd=fake_project, CENGBOT_NO_RESTART="0")
    assert proc.returncode == 0, out(proc)
    assert "[dry-run]" in proc.stdout
    assert snapshot(fake_project.parent) == before


def test_rollback_toggles_between_models(fake_project):
    assert run(SWITCH, "switch", "v2", root=fake_project).returncode == 0
    proc = run(SWITCH, "rollback", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert active(fake_project) == "final-best-model-v1.2"
    assert run(SWITCH, "rollback", root=fake_project).returncode == 0
    assert active(fake_project) == "final-best-model-v2"


def test_rollback_without_previous_fails(fake_project):
    proc = run(SWITCH, "rollback", root=fake_project)
    assert proc.returncode == 1 and "No previous model" in out(proc)
    assert active(fake_project) == "final-best-model-v1.2"


def test_rollback_refuses_path_outside_models(fake_project):
    (fake_project / "models" / ".previous-model").write_text("../../ext/new-model\n")
    before = snapshot(fake_project)
    proc = run(SWITCH, "rollback", root=fake_project)
    assert proc.returncode == 1
    assert snapshot(fake_project) == before


def test_import_from_outside_needs_confirmation(fake_project):
    src = fake_project.parent / "ext" / "new-model"
    before = snapshot(fake_project)
    proc = run(SWITCH, "switch", src, root=fake_project)          # stdin is not a tty -> "no"
    assert proc.returncode != 0 and "Cancelled" in out(proc)
    assert snapshot(fake_project) == before


def test_import_from_outside_copies_and_switches(fake_project, tmp_path):
    src = tmp_path / "ext" / "new-model"
    proc = run(SWITCH, "switch", "../ext/new-model", "--yes", root=fake_project, cwd=fake_project)
    assert proc.returncode == 0, out(proc)
    dest = fake_project / "models" / "new-model"
    assert (dest / "adapter_model.safetensors").read_bytes() == (src / "adapter_model.safetensors").read_bytes()
    assert src.is_dir(), "the source must be left in place (copy, not move)"
    assert active(fake_project) == "new-model"
    assert not list((fake_project / "models").glob(".importing-*"))
    # importing the same name again must not overwrite the existing model
    proc = run(SWITCH, "switch", src, "--yes", root=fake_project)
    assert proc.returncode != 0 and "already exists" in out(proc)
    proc = run(SWITCH, "switch", src, "--yes", "--name", "new-model-b", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert active(fake_project) == "new-model-b"


def test_import_move_on_same_filesystem(fake_project, tmp_path):
    proc = run(SWITCH, "switch", tmp_path / "ext" / "new-model", "--yes", "--move", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert not (tmp_path / "ext" / "new-model").exists()
    assert (fake_project / "models" / "new-model" / "adapter_config.json").is_file()


def test_import_rejects_invalid_outside_model(fake_project, tmp_path):
    bad = tmp_path / "ext" / "bad"
    bad.mkdir(parents=True)
    (bad / "adapter_config.json").write_text("{}")
    before = snapshot(fake_project)
    proc = run(SWITCH, "switch", bad, "--yes", root=fake_project)
    assert proc.returncode != 0
    assert snapshot(fake_project) == before


def test_refuses_to_replace_a_real_active_model_directory(fake_project):
    link = fake_project / "models" / "active-model"
    link.unlink()
    make_lora(link)
    before = snapshot(fake_project)
    proc = run(SWITCH, "switch", "v2", root=fake_project)
    assert proc.returncode != 0 and "NOT a symlink" in out(proc)
    assert snapshot(fake_project) == before


def test_works_from_any_directory(fake_project, tmp_path):
    proc = run(SWITCH, "switch", "v2", root=fake_project, cwd="/")
    assert proc.returncode == 0, out(proc)
    assert active(fake_project) == "final-best-model-v2"


def test_current_warns_about_model_path_override(fake_project):
    (fake_project / ".env").write_text("LORA_MODEL_PATH=models/final-best-model-v1/method1\nMODEL_PATH=models/full-merged\n")
    proc = run(SWITCH, "current", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert "MODEL_PATH=models/full-merged is set" in proc.stdout
    assert "LORA_MODEL_PATH" in proc.stdout and "ignored" in proc.stdout


def test_restart_hint_when_worker_not_running(fake_project):
    proc = run(SWITCH, "switch", "v2", root=fake_project, CENGBOT_NO_RESTART="0")
    assert proc.returncode == 0, out(proc)
    assert "AI worker is not running" in proc.stdout


# --------------------------------------------------------------------------
# process handling with dummy services (only processes started by the test)
# --------------------------------------------------------------------------
DUMMY_SERVICE = """
import os, signal, sys, time
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
print("dummy started", os.getpid(), flush=True)
while True:
    time.sleep(0.2)
"""


def spawn_dummy(cwd, script_rel):
    script = Path(cwd) / script_rel
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(DUMMY_SERVICE)
    proc = subprocess.Popen([sys.executable, script_rel], cwd=str(cwd),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # zombies count as dead
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def wait_dead(pid, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if not alive(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def cleanup_pids():
    pids = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_switch_restarts_only_this_projects_worker(fake_project, tmp_path, cleanup_pids):
    worker = spawn_dummy(fake_project, "src/ai_model_worker.py")
    bot = spawn_dummy(fake_project, "src/telegram_bot_rabbitmq.py")
    decoy = spawn_dummy(tmp_path / "other_project", "src/ai_model_worker.py")
    cleanup_pids.extend([worker.pid, bot.pid, decoy.pid])
    time.sleep(0.5)
    proc = run(SWITCH, "switch", "v2", "--yes", root=fake_project, CENGBOT_NO_RESTART="0")
    assert proc.returncode == 0, out(proc)
    worker.wait(timeout=10)
    new_pid = int((fake_project / ".pids").read_text().split()[0])
    cleanup_pids.append(new_pid)
    assert alive(new_pid) and new_pid != worker.pid
    assert "ai_model_worker.py" in Path(f"/proc/{new_pid}/cmdline").read_text()
    assert os.readlink(f"/proc/{new_pid}/cwd") == str(fake_project.resolve())
    assert alive(bot.pid), "the bot must keep running"
    assert alive(decoy.pid), "a worker of another project must not be touched"
    time.sleep(0.5)
    assert "dummy started" in (fake_project / "logs" / "worker.log").read_text()


def _holds_switch_lock(pid, root):
    lock = str((root / "models" / ".switch.lock").resolve())
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False
    for fd in fds:
        try:
            if os.readlink(f"/proc/{pid}/fd/{fd}") == lock:
                return True
        except OSError:
            continue
    return False


def test_restarted_worker_does_not_inherit_switch_lock(fake_project, cleanup_pids):
    """Regression: the worker started by `switch` inherited the flock fd, so every
    later switch/rollback failed with "Another switch_model.sh is running"."""
    worker = spawn_dummy(fake_project, "src/ai_model_worker.py")
    cleanup_pids.append(worker.pid)
    time.sleep(0.5)
    for args in (("switch", "v2"), ("rollback",), ("switch", "v2")):
        proc = run(SWITCH, *args, "--yes", root=fake_project, CENGBOT_NO_RESTART="0")
        assert proc.returncode == 0, out(proc)
        assert "Another switch_model.sh is running" not in proc.stdout + proc.stderr
        new_pid = int((fake_project / ".pids").read_text().split()[0])
        cleanup_pids.append(new_pid)
        assert alive(new_pid)
        assert not _holds_switch_lock(new_pid, fake_project), "worker must not hold the switch lock"


def test_stop_system_stops_only_this_project(fake_project, tmp_path, cleanup_pids):
    worker = spawn_dummy(fake_project, "src/ai_model_worker.py")
    api = spawn_dummy(fake_project, "src/admin_rest_api.py")
    legacy = spawn_dummy(fake_project, "src/telegram_bot.py")
    decoy = spawn_dummy(tmp_path / "other_project", "src/ai_model_worker.py")
    cleanup_pids.extend([worker.pid, api.pid, legacy.pid, decoy.pid])
    # a stale .pids entry that now belongs to an unrelated process (PID reuse)
    (fake_project / ".pids").write_text(f"{decoy.pid} - - -\n")
    time.sleep(0.5)
    proc = run(SCRIPTS / "stop_system.sh", root=fake_project, timeout=60)
    assert proc.returncode == 0, out(proc)
    for p in (worker, api, legacy):
        p.wait(timeout=10)
    assert alive(decoy.pid), "unrelated process must survive"
    assert "no longer AI Worker" in proc.stdout
    assert not (fake_project / ".pids").exists()


# --------------------------------------------------------------------------
# admin panel (Vite): package.json contract and dev-server process matching
# --------------------------------------------------------------------------
FRONTEND = ROOT / "admin_frontend"
FE_BIN = "/srv/cengbot/admin_frontend/node_modules/.bin"


def frontend_regex_matches(cmdlines, root):
    """service_regex frontend from _common.sh applied to each command line."""
    script = 'source "$1"; shift; re=$(service_regex frontend); for c in "$@"; do [[ "$c" =~ $re ]] && echo 1 || echo 0; done'
    proc = subprocess.run(["bash", "-c", script, "_", str(SCRIPTS / "_common.sh"), *cmdlines],
                          env=script_env(root), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    return [flag == "1" for flag in proc.stdout.split()]


def test_admin_frontend_package_uses_vite_not_react_scripts():
    pkg = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    assert "react-scripts" not in deps
    assert "react-scripts" not in json.dumps(pkg)
    scripts = pkg["scripts"]
    for name in ("start", "build", "test"):
        assert scripts.get(name), f"admin_frontend: npm run {name} is missing"
    assert scripts["start"] == "vite"  # _common.sh start_service frontend runs `npm start`
    assert "vite build" in scripts["build"]
    assert "vitest run" in scripts["test"]  # non-watch: `npm test` must terminate
    config = (FRONTEND / "vite.config.ts").read_text(encoding="utf-8")
    assert re.search(r"outDir:\s*['\"]build['\"]", config), "deploy_production.sh copies admin_frontend/build/"
    assert re.search(r"port:\s*3000\b", config), "the panel origin localhost:3000 is in the default CORS_ORIGINS"
    assert "admin_frontend/build/." in (SCRIPTS / "deploy_production.sh").read_text(encoding="utf-8")


@pytest.mark.parametrize("cmdline", [
    "npm start",
    "npm run start",
    "sh -c vite",
    f"node {FE_BIN}/vite",
    f"node {FE_BIN}/vite --host 127.0.0.1",
    f"node {FE_BIN}/vite dev",
    f"node {FE_BIN}/vite serve --port 3000",
    "node /srv/cengbot/admin_frontend/node_modules/vite/bin/vite.js",
])
def test_frontend_regex_matches_vite_dev_server(cmdline, tmp_path):
    assert frontend_regex_matches([cmdline], tmp_path) == [True]


@pytest.mark.parametrize("cmdline", [
    "npm test",
    "npm run test:watch",
    "npm run build",
    "npm run preview",
    "sh -c vitest run",
    "sh -c vitest",
    f"node {FE_BIN}/vitest run",
    f"node {FE_BIN}/vitest",
    "node --conditions node /srv/cengbot/admin_frontend/node_modules/vitest/dist/workers/forks.js",
    "sh -c tsc --noEmit && vite build",
    f"node {FE_BIN}/vite build",
    f"node {FE_BIN}/vite preview",
    "sh -c vite preview",
    "vim /srv/cengbot/admin_frontend/vite.config.ts",
])
def test_frontend_regex_ignores_vitest_build_and_preview(cmdline, tmp_path):
    assert frontend_regex_matches([cmdline], tmp_path) == [False]


def test_frontend_regex_follows_package_scripts(tmp_path):
    """What npm actually spawns for each script (`sh -c <script>`)."""
    scripts = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))["scripts"]
    names = ["start", "build", "test", "preview"]
    got = frontend_regex_matches([f"sh -c {scripts[n]}" for n in names if n in scripts], tmp_path)
    assert got == [n == "start" for n in names if n in scripts]


FAKE_NODE_BIN = """#!/bin/sh
trap 'exit 0' TERM
while :; do sleep 0.2; done
"""


def test_stop_system_stops_vite_dev_server_but_not_vitest(fake_project):
    """npm start -> `sh -c vite` -> .bin/vite is stopped; a vitest run or a
    `vite build` in the same project is left alone."""
    fe = fake_project / "admin_frontend"
    bin_dir = fe / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    for name in ("vite", "vitest"):
        (bin_dir / name).write_text(FAKE_NODE_BIN)
        (bin_dir / name).chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def spawn(command):
        return subprocess.Popen(["sh", "-c", command], cwd=str(fe), env=env, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def tree(proc):
        """The `sh -c` process plus the fake script it runs (or became), once started."""
        for _ in range(50):
            try:
                own = Path(f"/proc/{proc.pid}/cmdline").read_bytes()
                children = Path(f"/proc/{proc.pid}/task/{proc.pid}/children").read_text().split()
            except OSError:
                own, children = b"", []
            if children or b"/.bin/" in own:
                return [proc.pid, *map(int, children)]
            time.sleep(0.1)
        raise AssertionError(f"fake process did not start: {proc.args}")

    procs = [spawn("vite"), spawn("vitest run"), spawn("vite build")]
    try:
        dev, vitest, build = (tree(p) for p in procs)
        proc = run(SCRIPTS / "stop_system.sh", root=fake_project, timeout=60)
        assert proc.returncode == 0, out(proc)
        assert "Stopping React Frontend" in proc.stdout
        procs[0].wait(timeout=10)
        assert all(wait_dead(pid) for pid in dev), "the Vite dev server must be stopped"
        assert all(alive(pid) for pid in vitest), "a vitest run must not be touched"
        assert all(alive(pid) for pid in build), "a vite build must not be touched"
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            p.wait(timeout=10)


# --------------------------------------------------------------------------
# log rotation
# --------------------------------------------------------------------------
def rotate(path, root, **env):
    return subprocess.run(
        ["bash", "-c", 'source "$1"; rotate_log "$2"', "_", str(SCRIPTS / "_common.sh"), str(path)],
        env=script_env(root, **env), capture_output=True, text=True, timeout=30,
    )


def test_rotate_log_keeps_tail_and_bounded_generations(tmp_path):
    log = tmp_path / "logs" / "worker.log"
    log.parent.mkdir()
    for i in range(7):
        log.write_bytes(b"x" * 2000 + f"END{i}".encode())
        proc = rotate(log, tmp_path, CENGBOT_LOG_MAX_BYTES=1000, CENGBOT_LOG_KEEP_BYTES=100)
        assert proc.returncode == 0, out(proc)
        assert log.stat().st_size == 0
    rotated = sorted(p.name for p in log.parent.iterdir() if p.name != "worker.log")
    assert rotated == [f"worker.log.{i}.gz" for i in range(1, 6)]   # same naming as logrotate, rotate 5
    newest = gzip.decompress((log.parent / "worker.log.1.gz").read_bytes())
    assert newest.endswith(b"END6") and b"log rotated" in newest
    assert len(newest.split(b"\n", 1)[1]) == 100                   # only the tail was kept


def test_rotate_log_leaves_small_logs_alone(tmp_path):
    log = tmp_path / "bot.log"
    log.write_text("small\n")
    assert rotate(log, tmp_path, CENGBOT_LOG_MAX_BYTES=1000).returncode == 0
    assert log.read_text() == "small\n" and not list(tmp_path.glob("bot.log.*"))


def test_rotate_logs_dir_skips_python_managed_logs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    for name in ("error.log", "system_monitor.log", "admin.log"):
        (logs / name).write_bytes(b"y" * 3000)
    proc = subprocess.run(
        ["bash", "-c", 'source "$1"; rotate_logs_dir "$2"', "_", str(SCRIPTS / "_common.sh"), str(logs)],
        env=script_env(tmp_path, CENGBOT_LOG_MAX_BYTES=1000), capture_output=True, text=True,
    )
    assert proc.returncode == 0, out(proc)
    assert (logs / "error.log").stat().st_size == 3000
    assert (logs / "system_monitor.log").stat().st_size == 3000
    assert (logs / "admin.log").stat().st_size == 0 and (logs / "admin.log.1.gz").exists()


def test_start_service_appends_instead_of_truncating():
    text = (SCRIPTS / "_common.sh").read_text()
    assert '>> "$log" 2>&1' in text
    for script in ALL_SCRIPTS:
        assert not re.search(r"[^>]>\s*logs/\w+\.log", script.read_text()), f"{script.name} truncates a log"


def test_logrotate_config_directives():
    text = LOGROTATE_TEMPLATE.read_text()
    body = text[text.index("{"):]
    for directive in ("copytruncate", "size 50M", "rotate 5", "compress", "missingok", "notifempty", "@SU_LINE@"):
        assert directive in body, directive
    for name in ("worker", "bot", "admin", "frontend", "monitor"):
        assert f"@PROJECT_DIR@/logs/{name}.log" in text
    assert "@PROJECT_DIR@/logs/error.log" not in text      # rotated by RotatingFileHandler
    assert "@PROJECT_DIR@/logs/*.log" not in text


def test_rotate_logs_print_config(fake_project):
    proc = run(SCRIPTS / "rotate_logs.sh", "--print-config", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert "@" not in proc.stdout.split("{", 1)[1]
    assert f"{fake_project.resolve()}/logs/worker.log" in proc.stdout
    assert not re.search(r"^\s*su ", proc.stdout, re.M)
    proc = run(SCRIPTS / "rotate_logs.sh", "--print-config", "--system", root=fake_project,
               CENGBOT_SERVICE_USER=os.environ.get("USER") or "root")
    assert re.search(r"^\s*su \S+ \S+$", proc.stdout, re.M)


@pytest.mark.skipif(not (shutil.which("logrotate") or os.path.exists("/usr/sbin/logrotate")),
                    reason="logrotate not installed")
def test_logrotate_config_passes_debug_and_rotates(fake_project):
    logs = fake_project / "logs"
    big = logs / "worker.log"
    with open(big, "wb") as f:              # 51 MiB sparse file: above "size 50M"
        f.truncate(51 * 1024 * 1024)
    (logs / "bot.log").write_text("small\n")
    (logs / "error.log").write_bytes(b"e" * 10)
    before = snapshot(fake_project)
    proc = run(SCRIPTS / "rotate_logs.sh", "--debug", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert "error" not in out(proc).lower().replace("error.log", "")
    assert snapshot(fake_project) == before, "--debug must not change anything"
    proc = run(SCRIPTS / "rotate_logs.sh", "--quiet", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert big.stat().st_size == 0 and (logs / "worker.log.1.gz").exists()
    assert (logs / "bot.log").read_text() == "small\n"
    assert (logs / "error.log").stat().st_size == 10


def test_rotate_logs_pretrims_huge_logs_without_copying_them(fake_project):
    """A multi-GB log is trimmed with tail before logrotate runs, so copytruncate never
    has to copy the whole file (works with or without logrotate installed)."""
    logs = fake_project / "logs"
    huge = logs / "worker.log"
    with open(huge, "wb") as f:              # 1.5 GiB sparse file
        f.truncate(1536 * 1024 * 1024)
        f.seek(-6, 2)
        f.write(b"TAIL!\n")
    proc = run(SCRIPTS / "rotate_logs.sh", "--quiet", root=fake_project,
               CENGBOT_LOG_KEEP_BYTES=str(1024 * 1024))
    assert proc.returncode == 0, out(proc)
    assert huge.stat().st_size < 50 * 1024 * 1024
    rotated = sorted(logs.glob("worker.log.*.gz"))
    assert rotated, "the trimmed tail must be kept as a .gz generation"
    with gzip.open(rotated[0], "rb") as g:
        data = g.read()
    assert data.endswith(b"TAIL!\n") and len(data) < 2 * 1024 * 1024


# --------------------------------------------------------------------------
# other wrappers (fake project + stub Python modules)
# --------------------------------------------------------------------------
STUB_TRAIN = r"""
import json, os, sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
(root / "argv.json").write_text(json.dumps(sys.argv[1:]))
if os.environ.get("STUB_EXIT"):
    print("stub failing"); sys.exit(int(os.environ["STUB_EXIT"]))
if "--dry-run" not in sys.argv:
    d = root / "models" / "final-best-model-v2"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": "x/y"}))
    hdr = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    import struct
    (d / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr + b"\0\0\0\0")
    if not os.environ.get("STUB_NO_FINAL_LINE"):
        print(f"FINAL_MODEL_DIR={d}")
print("stub done")
"""


@pytest.fixture
def train_project(fake_project, tmp_path):
    (fake_project / "src" / "train_model.py").write_text(STUB_TRAIN)
    (fake_project / "data").mkdir()
    (fake_project / "data" / "merged_qa_data.jsonl").write_text('{"q": 1}\n{"q": 2}\n')
    shutil.rmtree(fake_project / "models" / "final-best-model-v2")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake_smi = bindir / "nvidia-smi"
    fake_smi.write_text("#!/bin/sh\necho 'Fake GPU, 24576, 0'\n")
    fake_smi.chmod(0o755)
    return fake_project, f"{bindir}:{os.environ['PATH']}"


def test_train_model_dry_run_forwards_options(train_project, tmp_path):
    root, path = train_project
    (root / "mine.jsonl").write_text('{"a": 1}\n')
    proc = run(SCRIPTS / "train_model.sh", "--dry-run", "--dataset", "../mine.jsonl", "--epochs", "2",
               root=root, cwd=root / "data", PATH=path)
    assert proc.returncode == 0, out(proc)
    argv = json.loads((root / "argv.json").read_text())
    assert argv[0] == "--dry-run"
    assert argv[1:3] == ["--dataset", str(root / "data" / "../mine.jsonl")]   # absolute, caller-relative
    assert argv[3:] == ["--epochs", "2"]
    assert not (root / "models" / "final-best-model-v2").exists()


def test_train_model_rejects_invalid_dataset(train_project):
    root, path = train_project
    (root / "data" / "merged_qa_data.jsonl").write_text('{"ok": 1}\nnot json\n')
    proc = run(SCRIPTS / "train_model.sh", "--dry-run", root=root, PATH=path)
    assert proc.returncode != 0 and "invalid JSON at line 2" in out(proc)


@pytest.mark.parametrize("final_line", [True, False], ids=["FINAL_MODEL_DIR", "mtime-fallback"])
def test_train_model_full_run_validates_new_model(train_project, final_line):
    root, path = train_project
    extra = {} if final_line else {"STUB_NO_FINAL_LINE": "1"}
    proc = run(SCRIPTS / "train_model.sh", "--yes", root=root, PATH=path, **extra)
    assert proc.returncode == 0, out(proc)
    assert f"New model directory: {root.resolve()}/models/final-best-model-v2" in proc.stdout
    assert "satisfies the model contract" in proc.stdout
    assert "switch_model.sh switch" in proc.stdout
    assert active(root) == "final-best-model-v1.2", "training must never switch the model"
    assert list((root / "logs" / "training").glob("auto_training_*.log"))


def test_train_model_failure_is_reported(train_project):
    root, path = train_project
    proc = run(SCRIPTS / "train_model.sh", "--yes", root=root, PATH=path, STUB_EXIT=3)
    assert proc.returncode != 0
    assert "Training FAILED (exit code 3)" in out(proc)


STUB_EXPORT = r"""
import sys, time
from pathlib import Path
root = Path(__file__).resolve().parent.parent
(root / "excel").mkdir(exist_ok=True)
(root / "export_argv.txt").write_text(" ".join(sys.argv[1:]))
name = "cengbot_database_export_%s.xlsx" % time.strftime("%Y%m%d_%H%M%S")
p = root / "excel" / name
n = 0
while p.exists():
    n += 1
    p = root / "excel" / name.replace(".xlsx", "_%d.xlsx" % n)
p.write_bytes(b"x")
print("exported")
"""


def test_export_database_forwards_args_and_prunes(fake_project):
    (fake_project / "src" / "export_to_excel.py").write_text(STUB_EXPORT)
    for _ in range(3):
        proc = run(SCRIPTS / "export_database.sh", "--all-tables", root=fake_project, KEEP_EXPORTS=2)
        assert proc.returncode == 0, out(proc)
        time.sleep(0.02)
    assert (fake_project / "export_argv.txt").read_text() == "--all-tables"
    assert len(list((fake_project / "excel").glob("cengbot_database_export_*.xlsx"))) == 2
    assert not list((fake_project / "excel").glob("*.backup"))


def test_backup_creates_archive_without_secrets(fake_project):
    import sqlite3
    db = sqlite3.connect(fake_project / "university_bot.db")
    db.execute("create table t(x)")
    db.execute("insert into t values (42)")
    db.commit()
    db.close()
    (fake_project / ".env").write_text("TELEGRAM_BOT_TOKEN=secret\n")
    (fake_project / "data").mkdir()
    (fake_project / "data" / "a.jsonl").write_text("{}\n")
    (fake_project / "logs" / "worker.log").write_bytes(b"z" * 5000 + b"TAIL")
    proc = run(SCRIPTS / "backup_data.sh", root=fake_project, LOG_TAIL_BYTES=100)
    assert proc.returncode == 0, out(proc)
    archives = list((fake_project / "backups").glob("backup_*.tar.gz"))
    assert len(archives) == 1
    assert not list((fake_project / "backups").glob(".staging_*"))
    with tarfile.open(archives[0]) as tar:
        names = tar.getnames()
        assert "./university_bot.db" in names and "./data/a.jsonl" in names
        assert "./models/active-model" in names and tar.getmember("./models/active-model").issym()
        assert not any(n.endswith(".env") for n in names)
        tail = tar.extractfile("./logs/worker.log.tail").read()
        assert tail.endswith(b"TAIL") and len(tail) == 100


def test_cleanup_only_touches_project_code_dirs(fake_project):
    (fake_project / "src" / "__pycache__").mkdir()
    (fake_project / "src" / "__pycache__" / "a.pyc").write_bytes(b"")
    venv_cache = fake_project / ".venv" / "lib" / "__pycache__"
    venv_cache.mkdir(parents=True)
    (venv_cache / "b.pyc").write_bytes(b"")
    (fake_project / "logs" / "bot.log").write_text("keep me\n")
    before = snapshot(fake_project)
    proc = run(SCRIPTS / "cleanup_system.sh", "--dry-run", "--clear-logs", root=fake_project)
    assert proc.returncode == 0, out(proc)
    assert snapshot(fake_project) == before
    proc = run(SCRIPTS / "cleanup_system.sh", "--clear-logs", root=fake_project)   # no tty -> logs kept
    assert not (fake_project / "src" / "__pycache__").exists()
    assert (venv_cache / "b.pyc").exists()
    assert (fake_project / "logs" / "bot.log").read_text() == "keep me\n"
    proc = run(SCRIPTS / "cleanup_system.sh", "--clear-logs", "--yes", root=fake_project)
    assert (fake_project / "logs" / "bot.log").read_text() == ""


def test_view_training_history_rejects_path_versions(fake_project):
    hist = fake_project / "logs" / "training_history"
    hist.mkdir(parents=True)
    (hist / "v1.log").write_text("line\n")
    (hist / "v1_info.json").write_text(json.dumps({"version": "v1", "final_loss": 0.8}))
    proc = run(SCRIPTS / "view_training_history.sh", "list", root=fake_project)
    assert proc.returncode == 0 and "Loss: 0.8" in proc.stdout, out(proc)
    proc = run(SCRIPTS / "view_training_history.sh", "view", "../../../etc/passwd", root=fake_project)
    assert proc.returncode == 1 and "Invalid version" in out(proc)
    proc = run(SCRIPTS / "view_training_history.sh", "best", root=fake_project)
    assert "Best performing model: v1" in proc.stdout


@pytest.mark.parametrize("script,args", [
    ("optimize_linux.sh", ["--dry-run"]),
    ("deploy_production.sh", ["--dry-run"]),
    ("backup_data.sh", ["--dry-run"]),
])
def test_system_scripts_dry_run_change_nothing(fake_project, script, args):
    venv = fake_project / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    before = snapshot(fake_project.parent)
    proc = run(SCRIPTS / script, *args, root=fake_project, CENGBOT_DEPLOY_BACKUP_DIR=fake_project.parent / "bk",
               CENGBOT_SYSTEMD_DIR=fake_project.parent / "systemd")
    assert proc.returncode == 0, out(proc)
    assert "dry" in proc.stdout.lower()
    assert snapshot(fake_project.parent) == before


def test_deploy_checks_node_before_stopping_anything(fake_project):
    """Vite needs Node 22.13+/24+: the deploy must fail before services are stopped,
    not after (which would leave production down)."""
    venv = fake_project / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    proc = run(SCRIPTS / "deploy_production.sh", "--dry-run", root=fake_project,
               CENGBOT_DEPLOY_BACKUP_DIR=fake_project.parent / "bk",
               CENGBOT_SYSTEMD_DIR=fake_project.parent / "systemd")
    assert proc.returncode == 0, out(proc)
    text = proc.stdout
    pre = text.find("Checking build prerequisites")
    assert pre != -1, out(proc)
    for later in ("backup", "Stopping"):
        idx = text.lower().find(later.lower(), pre)
        assert idx == -1 or idx > pre
    body = (SCRIPTS / "deploy_production.sh").read_text(encoding="utf-8")
    main = body[body.index("main() {"):]
    assert main.index("preflight_build_tools") < main.index("create_backup") < main.index("stop_services")


@pytest.mark.parametrize("script", ["optimize_linux.sh", "deploy_production.sh", "cleanup_system.sh",
                                    "setup_system.sh", "train_model.sh", "switch_model.sh"])
def test_dangerous_scripts_require_confirmation(script):
    text = (SCRIPTS / script).read_text()
    assert "confirm " in text or "confirm\n" in text, f"{script} must ask before acting"


def test_health_check_on_fake_project_is_read_only(fake_project):
    (fake_project / "logs" / "worker.log").write_text("w\n")
    before = snapshot(fake_project)
    proc = run(SCRIPTS / "health_check.sh", root=fake_project, timeout=60)
    assert "Active model" in proc.stdout
    assert "models/active-model -> final-best-model-v1.2 (valid)" in proc.stdout
    assert snapshot(fake_project) == before

"""
scripts/switch_model.sh (validate / switch) and scripts/train_model.sh with
non-Llama base models: Qwen, a DeepSeek-R1 distill and Mistral adapters in a
FAKE project (CENGBOT_ROOT=<tmp>). The base-model cache check works for any
Hugging Face org/name (``models--<org>--<name>``), honours MODEL_CACHE_DIR,
explains what to do when the base is not cached, and nothing assumes Llama.

Never touches the real models/, model_cache/ or logs/, never restarts anything.
"""

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SWITCH = SCRIPTS / "switch_model.sh"

QWEN = "Qwen/Qwen2.5-7B-Instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
BASES = {"qwen-v1": (QWEN, "qwen2"), "deepseek-v1": (DEEPSEEK, "qwen2"), "mistral-v1": (MISTRAL, "mistral")}

# Variables that change the scripts' behaviour (the real .env may have put some
# of them into os.environ during the test session).
_ENV_TO_CLEAR = ("MODEL_PATH", "ASSUME_YES", "BASE_MODEL_NAME", "MODEL_CACHE_DIR", "MODEL_TRUST_REMOTE_CODE",
                 "HUGGING_FACE_TOKEN", "HF_TOKEN")


def run(script, *args, root, cwd=None, **env):
    full_env = dict(os.environ, CENGBOT_ROOT=str(root), CENGBOT_PYTHON=sys.executable, CENGBOT_NO_RESTART="1",
                    NO_COLOR="1")
    for name in _ENV_TO_CLEAR:
        full_env.pop(name, None)
    full_env.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", str(script), *map(str, args)], cwd=str(cwd or root), env=full_env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)


def out(proc):
    return proc.stdout + proc.stderr


def write_safetensors(path):
    header = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    Path(path).write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")


def make_lora(directory: Path, base=None):
    directory.mkdir(parents=True, exist_ok=True)
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM"}
    if base is not None:
        cfg["base_model_name_or_path"] = base
    (directory / "adapter_config.json").write_text(json.dumps(cfg))
    write_safetensors(directory / "adapter_model.safetensors")
    (directory / "tokenizer.json").write_text("{}")
    return directory


def cache_base(cache: Path, repo: str, model_type: str, weights=True, auto_map=False, sharded=False):
    """Fake Hugging Face cache entry: models--<org>--<name>/snapshots/<rev>/ + refs/main."""
    repo_dir = cache / ("models--" + repo.replace("/", "--"))
    snap = repo_dir / "snapshots" / "0123abcd"
    snap.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("0123abcd")
    cfg = {"model_type": model_type, "architectures": ["X"]}
    if auto_map:
        cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}
    (snap / "config.json").write_text(json.dumps(cfg))
    (snap / "tokenizer.json").write_text("{}")
    if weights:
        if sharded:
            shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict(zip("ab", shards))}))
            for shard in shards:
                write_safetensors(snap / shard)
        else:
            write_safetensors(snap / "model.safetensors")
    return snap


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    for name, (base, _) in BASES.items():
        make_lora(root / "models" / name, base)
    os.symlink("mistral-v1", root / "models" / "active-model")
    (root / "logs").mkdir()
    return root


# ---------------------------------------------------------------------------
# validate: base model cache check for any org/name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(BASES))
def test_validate_accepts_cached_non_llama_bases(project, name):
    base, model_type = BASES[name]
    cache_base(project / "model_cache", base, model_type, sharded=(name == "deepseek-v1"))
    proc = run(SWITCH, "validate", name, root=project)
    assert proc.returncode == 0, out(proc)
    assert f"valid - LoRA adapter, base model: {base} (architecture: {model_type})" in proc.stdout
    assert "[WARNING]" not in out(proc)


@pytest.mark.parametrize("name", sorted(BASES))
def test_validate_explains_an_uncached_base(project, name):
    base, _ = BASES[name]
    proc = run(SWITCH, "validate", name, root=project)
    assert proc.returncode == 0, out(proc)  # a warning, not an error: the first load may download it
    text = out(proc)
    expected_dir = project / "model_cache" / ("models--" + base.replace("/", "--"))
    assert f"base model {base} is not cached in model_cache" in text
    assert str(expected_dir) in text
    assert "needs internet access to huggingface.co once" in text
    assert "MODEL_ALLOW_DOWNLOAD" in text and "HUGGING_FACE_TOKEN" in text


def test_validate_reports_incomplete_cached_weights(project):
    snap = cache_base(project / "model_cache", QWEN, "qwen2", sharded=True)
    (snap / "model-00002-of-00002.safetensors").unlink()  # interrupted download
    proc = run(SWITCH, "validate", "qwen-v1", root=project)
    assert proc.returncode == 0
    assert "no snapshot has complete weight files" in proc.stdout
    assert "not cached" not in proc.stdout


@pytest.mark.parametrize("relative", [True, False])
def test_validate_uses_model_cache_dir(project, tmp_path, relative):
    cache = tmp_path / "shared-hf-cache"
    cache_base(cache, DEEPSEEK, "qwen2")
    value = os.path.relpath(cache, project) if relative else str(cache)
    proc = run(SWITCH, "validate", "deepseek-v1", root=project, MODEL_CACHE_DIR=value)
    assert proc.returncode == 0, out(proc)
    assert "not cached" not in out(proc) and "(architecture: qwen2)" in proc.stdout
    # also read from the project's .env
    (project / ".env").write_text(f"MODEL_CACHE_DIR={cache}\n")
    proc = run(SWITCH, "validate", "deepseek-v1", root=project)
    assert "not cached" not in out(proc)


def test_validate_has_no_built_in_llama_default(project):
    make_lora(project / "models" / "nobase", base=None)
    proc = run(SWITCH, "validate", "nobase", root=project)
    assert proc.returncode == 0, out(proc)
    assert "has no base_model_name_or_path and BASE_MODEL_NAME is not set" in proc.stdout
    assert "will be used" not in proc.stdout and proc.stdout.count("[WARNING]") == 1
    assert "meta-llama" not in out(proc).lower() and "llama" not in out(proc).lower()

    cache_base(project / "model_cache", MISTRAL, "mistral")
    proc = run(SWITCH, "validate", "nobase", root=project, BASE_MODEL_NAME=MISTRAL)
    assert proc.returncode == 0
    assert f"base model: {MISTRAL} (architecture: mistral)" in proc.stdout
    assert "BASE_MODEL_NAME (mistralai/Mistral-7B-Instruct-v0.3) will be used" in proc.stdout


def test_validate_flags_remote_code_and_unknown_architectures(project):
    legacy = "THUDM/chatglm3-6b"
    make_lora(project / "models" / "chatglm3-v1", legacy)
    cache_base(project / "model_cache", legacy, "chatglm", auto_map=True)
    proc = run(SWITCH, "validate", "chatglm3-v1", root=project)
    assert proc.returncode == 0
    assert "MODEL_TRUST_REMOTE_CODE=true" in proc.stdout and "custom code" in proc.stdout
    proc = run(SWITCH, "validate", "chatglm3-v1", root=project, MODEL_TRUST_REMOTE_CODE="true")
    assert "MODEL_TRUST_REMOTE_CODE" not in proc.stdout

    make_lora(project / "models" / "mystery-v1", "acme/mystery-1b")
    cache_base(project / "model_cache", "acme/mystery-1b", "mystery_arch")
    proc = run(SWITCH, "validate", "mystery-v1", root=project)
    assert "architecture 'mystery_arch' is not supported by the installed transformers" in proc.stdout

    # GLM-4 -hf / 0414 checkpoints are native: no remote-code hint
    make_lora(project / "models" / "glm4-v1", "THUDM/GLM-4-9B-0414")
    cache_base(project / "model_cache", "THUDM/GLM-4-9B-0414", "glm4")
    proc = run(SWITCH, "validate", "glm4-v1", root=project)
    assert proc.returncode == 0 and "[WARNING]" not in out(proc)


def test_validate_local_base_paths(project, tmp_path):
    base_dir = tmp_path / "bases" / "qwen-local"
    base_dir.mkdir(parents=True)
    (base_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    make_lora(project / "models" / "local-v1", str(base_dir))
    proc = run(SWITCH, "validate", "local-v1", root=project)
    assert proc.returncode == 0 and "[WARNING]" not in out(proc)
    assert "(architecture: qwen2)" in proc.stdout

    make_lora(project / "models" / "gone-v1", "/nonexistent/base/model")
    proc = run(SWITCH, "validate", "gone-v1", root=project)
    assert proc.returncode == 0
    assert "base model path /nonexistent/base/model does not exist on this machine" in proc.stdout


def test_validate_full_model_with_bpe_tokenizer_files(project):
    full = project / "models" / "merged-qwen"
    full.mkdir()
    (full / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    write_safetensors(full / "model.safetensors")
    (full / "vocab.json").write_text("{}")
    (full / "merges.txt").write_text("#version: 0.2\n")
    proc = run(SWITCH, "validate", "merged-qwen", root=project)
    assert proc.returncode == 0, out(proc)
    assert "valid - full model (architecture: qwen2)" in proc.stdout
    (full / "merges.txt").unlink()
    proc = run(SWITCH, "validate", "merged-qwen", root=project)
    assert proc.returncode == 1 and "no tokenizer files" in out(proc)


def test_switch_to_a_qwen_adapter_and_back(project):
    cache_base(project / "model_cache", QWEN, "qwen2")
    proc = run(SWITCH, "switch", "qwen-v1", root=project)
    assert proc.returncode == 0, out(proc)
    assert os.readlink(project / "models" / "active-model") == "qwen-v1"
    assert (project / "models" / ".previous-model").read_text().strip() == "mistral-v1"
    proc = run(SWITCH, "rollback", root=project)
    assert proc.returncode == 0, out(proc)
    assert os.readlink(project / "models" / "active-model") == "mistral-v1"


def test_list_shows_every_family(project):
    proc = run(SWITCH, "list", root=project)
    assert proc.returncode == 0, out(proc)
    for name in BASES:
        assert name in proc.stdout
    assert "INVALID" not in proc.stdout


def test_switch_model_script_has_no_llama_assumptions():
    script = SWITCH.read_text(encoding="utf-8")
    assert "Llama-3.2" not in script  # no built-in default base model
    assert "models--meta-llama" not in script  # no hardcoded cache directory


# ---------------------------------------------------------------------------
# train_model.sh: the base-model cache hint follows --base-model / BASE_MODEL_NAME
# ---------------------------------------------------------------------------
STUB_TRAIN = r"""
import json, sys
from pathlib import Path
(Path(__file__).resolve().parent.parent / "argv.json").write_text(json.dumps(sys.argv[1:]))
print("stub done")
"""


@pytest.fixture
def train_project(project):
    (project / "src").mkdir()
    (project / "src" / "train_model.py").write_text(STUB_TRAIN)
    (project / "data").mkdir()
    (project / "data" / "merged_qa_data.jsonl").write_text('{"q": 1}\n')
    return project


def test_train_wrapper_warns_for_uncached_non_llama_base(train_project):
    proc = run(SCRIPTS / "train_model.sh", "--dry-run", "--base-model", QWEN, "--load-in-4bit", root=train_project)
    assert proc.returncode == 0, out(proc)
    assert f"Base model {QWEN} is not cached" in proc.stdout
    assert "downloads it once" in proc.stdout and "HUGGING_FACE_TOKEN is empty" in proc.stdout
    argv = json.loads((train_project / "argv.json").read_text())
    assert argv == ["--dry-run", "--base-model", QWEN, "--load-in-4bit"]


def test_train_wrapper_quiet_when_base_is_cached(train_project, tmp_path):
    cache_base(train_project / "model_cache", MISTRAL, "mistral")
    proc = run(SCRIPTS / "train_model.sh", "--dry-run", root=train_project, BASE_MODEL_NAME=MISTRAL)
    assert proc.returncode == 0, out(proc)
    assert "not cached" not in proc.stdout
    # --base-model=... wins over BASE_MODEL_NAME; MODEL_CACHE_DIR is honoured
    cache = tmp_path / "other-cache"
    cache_base(cache, DEEPSEEK, "qwen2")
    proc = run(SCRIPTS / "train_model.sh", "--dry-run", f"--base-model={DEEPSEEK}", root=train_project,
               BASE_MODEL_NAME=MISTRAL, MODEL_CACHE_DIR=cache, HUGGING_FACE_TOKEN="hf_dummy")
    assert proc.returncode == 0 and "not cached" not in proc.stdout
    # a local base directory never needs a download
    local = tmp_path / "local-base"
    local.mkdir()
    proc = run(SCRIPTS / "train_model.sh", "--dry-run", "--base-model", local, root=train_project)
    assert proc.returncode == 0 and "not cached" not in proc.stdout

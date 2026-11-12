"""
Real end-to-end training smoke test on the GPU (opt-in).

Skipped unless RUN_GPU_TESTS=1. Uses the real base model from model_cache/
(offline) and a tiny stratified TR/EN sample of data/merged_qa_data.jsonl:

  scripts/train_model.sh (3 optimizer steps, all outputs in tmp_path)
  -> switch_model.sh validate (run by the wrapper)
  -> scripts/view_training_history.sh --log-dir
  -> the production handler (MODEL_PATH=<new model>) answers a TR and an EN question

Nothing is written to the project's models/, checkpoints/ or logs/. Run it while
holding the GPU so that no other job shares it, e.g.
``flock <gpu.lock> env RUN_GPU_TESTS=1 .venv/bin/python -m pytest -q tests/test_training_e2e_gpu.py``.
Takes about a minute on an RTX 3090.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
BASE = "meta-llama/Llama-3.2-3B"
DATA = ROOT / "data" / "merged_qa_data.jsonl"
CACHE = ROOT / "model_cache"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1 to train on the GPU"),
]

GENERATE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import llama_model_handler as h
m = h.model_instance
if not m.load_model():
    print(json.dumps({"error": str(m.last_load_error)})); sys.exit(1)
answers = {q: m.generate_response(q) for q in ("Staj zorunlu mu?", "Is the internship mandatory?")}
print("RESULT=" + json.dumps({"answers": answers, "info": m.get_model_info()}, ensure_ascii=False, default=str))
"""


def _env(**extra):
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_COLOR="1", ASSUME_YES="1")
    for name in ("MODEL_PATH", "MODEL_CACHE_DIR", "TRAINING_HISTORY_DIR", "TRAINING_VERSION"):
        env.pop(name, None)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _sample(path: Path, out: Path, per_language: int = 10) -> Path:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(0)
    picked = []
    for lang in ("turkish", "english"):
        picked += rng.sample([r for r in rows if r.get("language") == lang], per_language)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked), encoding="utf-8")
    return out


def test_real_training_run_produces_a_model_the_handler_can_serve(tmp_path):
    import torch

    import train_model as tm

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not DATA.is_file():
        pytest.skip(f"{DATA} not found")
    if not tm.is_model_cached(BASE, str(CACHE)):
        pytest.skip(f"{BASE} is not cached in {CACHE}")
    project_state = {p: p.exists() for p in (ROOT / "checkpoints", ROOT / "logs" / "training_history",
                                             ROOT / "logs" / "training")}
    models_before = sorted(os.listdir(ROOT / "models"))

    dataset = _sample(DATA, tmp_path / "sample.jsonl")
    models, ckpts, logs = tmp_path / "models", tmp_path / "ckpt", tmp_path / "logs"
    proc = subprocess.run(
        ["bash", str(SCRIPTS / "train_model.sh"), "--yes", "--dataset", str(dataset), "--models-dir", str(models),
         "--checkpoint-dir", str(ckpts), "--log-dir", str(logs), "--base-model", BASE, "--cache-dir", str(CACHE),
         "--no-db-export", "--max-steps", "3", "--batch-size", "2", "--grad-accum", "1", "--eval-batch-size", "4",
         "--evals-per-epoch", "3", "--logging-steps", "1", "--workers", "0"],
        cwd=str(tmp_path), env=_env(), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output[-4000:]
    final_dir = models / "final-best-model-v1"
    assert f"FINAL_MODEL_DIR={final_dir}" in output
    assert "satisfies the model contract" in output  # switch_model.sh validate, run by the wrapper
    assert "Could not find a config file" not in output and "TOKENIZERS_PARALLELISM" not in output

    info = json.loads((final_dir / "training_info.json").read_text(encoding="utf-8"))
    assert info["total_steps"] == 3 and info["max_steps"] == 3 and info["base_model"] == BASE
    assert info["precision"] == "bf16" and info["peak_gpu_memory_gb"] > 1
    assert info["train_examples"] + info["val_examples"] == 20
    assert info["metrics"]["final_eval_loss"] is not None and info["final_loss"] is not None
    assert {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "training_info.json"} <= {p.name for p in final_dir.iterdir()}
    assert list((ckpts / "final-best-model-v1").glob("checkpoint-*"))
    assert list((logs / "training").glob("auto_training_*.log"))

    history = subprocess.run(["bash", str(SCRIPTS / "view_training_history.sh"), "--log-dir", str(logs), "list"],
                             env=_env(), capture_output=True, text=True, timeout=60)
    assert history.returncode == 0 and "v1:" in history.stdout and "final-best-model-v1" in history.stdout

    serve = subprocess.run([sys.executable, "-c", GENERATE, str(ROOT / "src")], cwd=str(tmp_path),
                           env=_env(MODEL_PATH=final_dir, MODEL_CACHE_DIR=CACHE, MODEL_MAX_NEW_TOKENS=40),
                           capture_output=True, text=True, timeout=300)
    result_line = [line for line in serve.stdout.splitlines() if line.startswith("RESULT=")]
    assert serve.returncode == 0 and result_line, (serve.stdout + serve.stderr)[-4000:]
    result = json.loads(result_line[-1][len("RESULT="):])
    assert all(isinstance(a, str) and a.strip() for a in result["answers"].values()), result
    assert result["info"].get("version") == "final-best-model-v1"

    # the project's own directories were not touched
    assert {p: p.exists() for p in project_state} == project_state
    assert sorted(os.listdir(ROOT / "models")) == models_before

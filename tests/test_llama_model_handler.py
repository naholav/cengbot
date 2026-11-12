"""Tests for src/llama_model_handler.py.

The default suite never loads real weights: transformers/peft loaders are
replaced by mocks and generation uses a tiny character-level fake tokenizer /
model. The only real-model test is ``test_gpu_smoke_real_model_offline``
(marked ``gpu``, skipped unless RUN_GPU_TESTS=1).
"""

import ast
import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
import torch

import llama_model_handler as h

ROOT = Path(__file__).resolve().parent.parent
_REAL_DEFAULT_HF_CACHE = h._default_hf_cache
BASE_ID = "meta-llama/Llama-3.2-3B"

MODEL_ENV_VARS = [
    "BASE_MODEL_NAME", "MODEL_PATH", "LORA_MODEL_PATH", "MODEL_CACHE_DIR", "MODEL_TEMPERATURE",
    "MODEL_MAX_NEW_TOKENS", "MODEL_TOP_P", "MODEL_TOP_K", "MODEL_REPETITION_PENALTY", "MODEL_PRECISION",
    "USE_CUDA", "MODEL_MAX_INPUT_TOKENS", "MODEL_MAX_SENTENCES", "MODEL_PROMPT_STYLE", "MODEL_MERGE_LORA",
    "MODEL_ALLOW_DOWNLOAD", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HUGGING_FACE_TOKEN", "HF_TOKEN",
    "MODEL_TRUST_REMOTE_CODE", "MODEL_LOAD_IN_4BIT",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in MODEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(h, "_legacy_warning_emitted", False)
    # never write to the real logs/error.log, never look at the real ~/.cache
    monkeypatch.setattr(h, "handle_model_error", mock.Mock(name="handle_model_error"))
    monkeypatch.setattr(h, "_default_hf_cache", lambda: str(tmp_path / "no-default-hf-cache"))
    yield


# --------------------------------------------------------------------------
# Fake model directories
# --------------------------------------------------------------------------

def _write(path: Path, content="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def make_lora(root: Path, name="model-v2", layout="direct", base=BASE_ID, tokenizer=True, weights=True):
    model_dir = root / name
    adapter = model_dir / "method1" if layout == "method1" else model_dir
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16}
    if base is not None:
        cfg["base_model_name_or_path"] = base
    _write(adapter / "adapter_config.json", cfg)
    if weights:
        _write(adapter / "adapter_model.safetensors")
    if tokenizer:
        _write(adapter / "tokenizer.json", {})
        _write(adapter / "tokenizer_config.json", {})
    return model_dir


def make_full(root: Path, name="merged", tokenizer=True, shards=("model-00001-of-00002.safetensors",
                                                                  "model-00002-of-00002.safetensors"),
              missing_shard=False):
    model_dir = root / name
    _write(model_dir / "config.json", {"model_type": "llama"})
    _write(model_dir / "model.safetensors.index.json",
           {"weight_map": {f"layer{i}": shard for i, shard in enumerate(shards)}})
    for i, shard in enumerate(shards):
        if not (missing_shard and i == len(shards) - 1):
            _write(model_dir / shard)
    if tokenizer:
        _write(model_dir / "tokenizer.json", {})
    return model_dir


def make_cache(root: Path, repo=BASE_ID, main_ref=True, complete=True, tokenizer=True):
    cache = root / "model_cache"
    repo_dir = cache / ("models--" + repo.replace("/", "--"))
    snap = repo_dir / "snapshots" / "abc123"
    _write(snap / "config.json", {"model_type": "llama"})
    _write(snap / "model.safetensors.index.json",
           {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
    _write(snap / "model-00001-of-00002.safetensors")
    if complete:
        _write(snap / "model-00002-of-00002.safetensors")
    if tokenizer:
        _write(snap / "tokenizer.json", {})
    if main_ref:
        _write(repo_dir / "refs" / "main", "abc123")
    return cache, snap


@pytest.fixture
def lora_env(tmp_path, monkeypatch):
    """A fake LoRA model behind an active-model symlink plus a complete base model cache."""
    models = tmp_path / "models"
    target = make_lora(models, "final-best-model-v9")
    active = models / "active-model"
    active.symlink_to(target.name)
    cache, snap = make_cache(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(active))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(cache))
    return {"active": active, "target": target, "cache": cache, "snapshot": snap}


@pytest.fixture
def loaders(monkeypatch):
    """Mock transformers/peft loaders (no weights are ever read)."""
    tokenizer = mock.MagicMock(name="tokenizer")
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|end_of_text|>"
    tokenizer.eos_token_id = 128001
    tokenizer.unk_token_id = None
    llama3_specials = {"<|begin_of_text|>": 128000, "<|end_of_text|>": 128001, "<|eot_id|>": 128009}
    tokenizer.get_added_vocab.return_value = dict(llama3_specials)
    tokenizer.all_special_tokens = ["<|begin_of_text|>", "<|end_of_text|>"]
    tokenizer.convert_tokens_to_ids.side_effect = llama3_specials.get
    base_model = mock.MagicMock(name="base_model")
    peft_model = mock.MagicMock(name="peft_model")
    merged = mock.MagicMock(name="merged_model")
    peft_model.merge_and_unload.return_value = merged
    auto_tok = mock.MagicMock(name="AutoTokenizer")
    auto_tok.from_pretrained.return_value = tokenizer
    auto_model = mock.MagicMock(name="AutoModelForCausalLM")
    auto_model.from_pretrained.return_value = base_model
    peft = mock.MagicMock(name="PeftModel")
    peft.from_pretrained.return_value = peft_model
    monkeypatch.setattr(h, "AutoTokenizer", auto_tok)
    monkeypatch.setattr(h, "AutoModelForCausalLM", auto_model)
    monkeypatch.setattr(h, "PeftModel", peft)
    monkeypatch.setattr(h.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(h.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(h.torch.cuda, "empty_cache", lambda: None)
    return mock.Mock(tokenizer=tokenizer, base_model=base_model, peft_model=peft_model, merged=merged,
                     AutoTokenizer=auto_tok, AutoModelForCausalLM=auto_model, PeftModel=peft)


# --------------------------------------------------------------------------
# Config / env
# --------------------------------------------------------------------------

def test_model_config_defaults():
    cfg = h.ModelConfig()
    assert cfg.base_model == BASE_ID
    assert Path(cfg.model_path) == ROOT / "models" / "active-model"
    assert Path(cfg.cache_dir) == ROOT / "model_cache"
    assert (cfg.temperature, cfg.max_new_tokens, cfg.top_p, cfg.top_k, cfg.repetition_penalty) == \
        (0.7, 200, 0.95, 50, 1.1)
    assert cfg.precision == "bfloat16" and cfg.use_cuda is True
    assert cfg.prompt_style == "training"
    assert cfg.system_prompt_tr == h.TRAINING_SYSTEM_PROMPT_TR
    # class attributes stay available for backwards compatibility
    assert h.ModelConfig.temperature == 0.7 and h.ModelConfig.base_model == BASE_ID


def test_model_config_env_overrides(monkeypatch, tmp_path):
    env = {
        "BASE_MODEL_NAME": "org/other", "MODEL_PATH": "models/custom", "MODEL_CACHE_DIR": "/abs/cache",
        "MODEL_TEMPERATURE": "0.2", "MODEL_MAX_NEW_TOKENS": "64", "MODEL_TOP_P": "0.8", "MODEL_TOP_K": "10",
        "MODEL_REPETITION_PENALTY": "1.3", "MODEL_PRECISION": "float16", "USE_CUDA": "false",
        "MODEL_MAX_SENTENCES": "2", "MODEL_PROMPT_STYLE": "legacy", "MODEL_MERGE_LORA": "0",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)  # relative paths must not depend on the CWD
    cfg = h.ModelConfig()
    assert cfg.base_model == "org/other" and cfg.base_model_from_env
    assert Path(cfg.model_path) == ROOT / "models" / "custom"
    assert cfg.lora_model_path == cfg.model_path
    assert cfg.cache_dir == "/abs/cache"
    assert (cfg.temperature, cfg.max_new_tokens, cfg.top_p, cfg.top_k, cfg.repetition_penalty) == \
        (0.2, 64, 0.8, 10, 1.3)
    assert cfg.greeting_max_new_tokens == 30
    assert cfg.precision == "float16" and cfg.use_cuda is False
    assert cfg.max_sentences == 2 and cfg.merge_lora is False
    assert cfg.prompt_style == "legacy" and cfg.system_prompt_en == h.LEGACY_SYSTEM_PROMPT_EN
    assert cfg.turn_markers["tr"] == ("Student", "Assistant")


def test_invalid_env_values_fall_back(monkeypatch, caplog):
    monkeypatch.setenv("MODEL_TEMPERATURE", "hot")
    monkeypatch.setenv("MODEL_TOP_K", "many")
    monkeypatch.setenv("MODEL_PROMPT_STYLE", "weird")
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "10")
    cfg = h.ModelConfig()
    assert cfg.temperature == 0.7 and cfg.top_k == 50 and cfg.prompt_style == "training"
    assert cfg.max_new_tokens == 10 and cfg.greeting_max_new_tokens == 10


def test_lora_model_path_is_ignored_with_one_time_warning(lora_env, monkeypatch, caplog):
    monkeypatch.setenv("LORA_MODEL_PATH", "models/final-best-model-v1/method1")
    cfg = h.ModelConfig()
    assert Path(cfg.model_path) == lora_env["active"]
    with caplog.at_level(logging.WARNING, logger=h.logger.name):
        h._warn_legacy_lora_path(cfg.model_path)
        h._warn_legacy_lora_path(cfg.model_path)
    warnings = [r for r in caplog.records if "LORA_MODEL_PATH" in r.getMessage()]
    assert len(warnings) == 1 and "IGNORED" in warnings[0].getMessage()
    assert h.resolve_model_spec(cfg).model_dir == str(lora_env["target"])


def test_no_legacy_warning_when_it_matches_active_model(lora_env, monkeypatch, caplog):
    monkeypatch.setenv("LORA_MODEL_PATH", str(lora_env["target"] / "method1"))
    with caplog.at_level(logging.WARNING, logger=h.logger.name):
        h._warn_legacy_lora_path(str(lora_env["active"]))
    assert not [r for r in caplog.records if "LORA_MODEL_PATH" in r.getMessage()]


# --------------------------------------------------------------------------
# Model directory contract
# --------------------------------------------------------------------------

def _cfg(monkeypatch, path, cache=None):
    monkeypatch.setenv("MODEL_PATH", str(path))
    if cache is not None:
        monkeypatch.setenv("MODEL_CACHE_DIR", str(cache))
    return h.ModelConfig()


def test_resolve_direct_lora_layout(lora_env):
    spec = h.resolve_model_spec(h.ModelConfig())
    assert spec.kind == "lora"
    assert spec.model_dir == str(lora_env["target"])
    assert spec.version == "final-best-model-v9"
    assert spec.base_model == BASE_ID
    # base model is cached -> hub id + cache dir, local only
    assert spec.base.name == BASE_ID and spec.base.cache_dir == str(lora_env["cache"]) and spec.base.cached
    # tokenizer from the model dir first, base tokenizer as fallback
    assert spec.tokenizer_source == str(lora_env["target"])
    assert spec.tokenizer_sources[-1].online_name == BASE_ID


def test_resolve_legacy_method1_layout(tmp_path, monkeypatch):
    model_dir = make_lora(tmp_path, "final-best-model-v1", layout="method1")
    cache, _ = make_cache(tmp_path)
    spec = h.resolve_model_spec(_cfg(monkeypatch, model_dir, cache))
    assert spec.kind == "lora"
    assert spec.model_dir == str(model_dir / "method1")
    assert spec.tokenizer_source == str(model_dir / "method1")
    assert spec.version == "final-best-model-v1"


def test_incomplete_top_level_adapter_falls_back_to_complete_method1(tmp_path, monkeypatch):
    """Same rule as `switch_model.sh validate`: a stray top-level adapter_config.json
    without weights must not hide a complete legacy method1/ adapter."""
    model_dir = make_lora(tmp_path, "final-best-model-v1", layout="method1")
    (model_dir / "adapter_config.json").write_text((model_dir / "method1" / "adapter_config.json").read_text())
    cache, _ = make_cache(tmp_path)
    spec = h.resolve_model_spec(_cfg(monkeypatch, model_dir, cache))
    assert spec.model_dir == str(model_dir / "method1")


def test_adapter_config_without_any_weights_is_rejected(tmp_path, monkeypatch):
    model_dir = make_lora(tmp_path, "broken")
    for f in model_dir.glob("adapter_model.*"):
        f.unlink()
    cache, _ = make_cache(tmp_path)
    with pytest.raises(h.ModelLoadError, match="weights are missing"):
        h.resolve_model_spec(_cfg(monkeypatch, model_dir, cache))


def test_resolve_adapter_without_tokenizer_uses_base_tokenizer(tmp_path, monkeypatch):
    model_dir = make_lora(tmp_path, tokenizer=False)
    cache, _ = make_cache(tmp_path)
    spec = h.resolve_model_spec(_cfg(monkeypatch, model_dir, cache))
    assert len(spec.tokenizer_sources) == 1
    assert spec.tokenizer_source == BASE_ID and spec.tokenizer_sources[0].cached


def test_resolve_full_model_layout(tmp_path, monkeypatch):
    model_dir = make_full(tmp_path)
    spec = h.resolve_model_spec(_cfg(monkeypatch, model_dir))
    assert spec.kind == "full" and spec.base_model is None
    assert spec.base.name == str(model_dir) and spec.base.cached and spec.base.online_name is None
    assert spec.tokenizer_source == str(model_dir)


@pytest.mark.parametrize("builder,needle", [
    (lambda root: make_full(root, missing_shard=True), "weights are missing or incomplete"),
    (lambda root: make_lora(root, weights=False), "adapter weights are missing"),
    (lambda root: _write(root / "empty" / "notes.txt").parent, "not a valid model directory"),
    (lambda root: root / "does-not-exist", "Model directory not found"),
])
def test_resolve_validation_errors(tmp_path, monkeypatch, builder, needle):
    path = builder(tmp_path)
    with pytest.raises(h.ModelLoadError, match=needle):
        h.resolve_model_spec(_cfg(monkeypatch, path))


def test_resolve_broken_symlink(tmp_path, monkeypatch):
    link = tmp_path / "active-model"
    link.symlink_to(tmp_path / "gone")
    with pytest.raises(h.ModelLoadError, match="broken symlink"):
        h.resolve_model_spec(_cfg(monkeypatch, link))


def test_adapter_without_base_uses_base_model_name(tmp_path, monkeypatch):
    model_dir = make_lora(tmp_path, base=None)
    monkeypatch.setenv("BASE_MODEL_NAME", "org/fallback-base")
    spec = h.resolve_model_spec(_cfg(monkeypatch, model_dir))
    assert spec.base_model == "org/fallback-base"


def test_adapter_with_foreign_local_base_path_falls_back(tmp_path, monkeypatch, caplog):
    model_dir = make_lora(tmp_path, base="/home/someone-else/llama-3.2-3b")
    with caplog.at_level(logging.WARNING, logger=h.logger.name):
        spec = h.resolve_model_spec(_cfg(monkeypatch, model_dir))
    assert spec.base_model == BASE_ID
    assert any("does not exist here" in r.getMessage() for r in caplog.records)


def test_base_model_mismatch_warns_and_uses_adapter_base(tmp_path, monkeypatch, caplog):
    model_dir = make_lora(tmp_path, base="meta-llama/Llama-3.2-3B-Instruct")
    monkeypatch.setenv("BASE_MODEL_NAME", BASE_ID)
    with caplog.at_level(logging.WARNING, logger=h.logger.name):
        spec = h.resolve_model_spec(_cfg(monkeypatch, model_dir))
    assert spec.base_model == "meta-llama/Llama-3.2-3B-Instruct"
    assert any("differs from the adapter's base model" in r.getMessage() for r in caplog.records)


def test_cache_lookup_variants(tmp_path):
    cache, snap = make_cache(tmp_path, main_ref=False)
    found = h.find_cached_snapshot(BASE_ID, [str(cache)])
    assert found is not None and found[1] == snap and found[2] is False
    cfg = h.ModelConfig()
    cfg.cache_dir = str(cache)
    source = h._hub_source(BASE_ID, cfg, need_weights=True, need_tokenizer=False, what="Base model")
    assert source.name == str(snap) and source.cache_dir is None and source.cached  # snapshot dir directly

    cache2, _ = make_cache(tmp_path / "other", complete=False)
    assert h.find_cached_snapshot(BASE_ID, [str(cache2)]) is None  # missing shard -> not cached
    cfg.cache_dir = str(cache2)
    source = h._hub_source(BASE_ID, cfg, need_weights=True, need_tokenizer=False, what="Base model")
    assert not source.cached and source.cache_dir == str(cache2) and source.online_name == BASE_ID


# --------------------------------------------------------------------------
# load_model (mocked loaders)
# --------------------------------------------------------------------------

def test_load_model_is_offline_first_and_safe(lora_env, loaders):
    model = h.CengBotModel()
    assert model.load_model() is True

    tok_call = loaders.AutoTokenizer.from_pretrained.call_args_list[0]
    assert tok_call.args == (str(lora_env["target"]),)
    assert tok_call.kwargs["local_files_only"] is True and "token" not in tok_call.kwargs

    base_call = loaders.AutoModelForCausalLM.from_pretrained.call_args_list[0]
    assert base_call.args == (BASE_ID,)
    assert base_call.kwargs["local_files_only"] is True
    assert base_call.kwargs["cache_dir"] == str(lora_env["cache"])
    assert base_call.kwargs["torch_dtype"] == torch.bfloat16
    assert base_call.kwargs["device_map"] == "auto"
    assert "token" not in base_call.kwargs

    peft_call = loaders.PeftModel.from_pretrained.call_args_list[0]
    assert peft_call.args == (loaders.base_model, str(lora_env["target"]))
    assert peft_call.kwargs["local_files_only"] is True and peft_call.kwargs["is_trainable"] is False

    for call in (tok_call, base_call, peft_call):
        assert "trust_remote_code" not in call.kwargs
    assert loaders.AutoTokenizer.from_pretrained.call_count == 1
    assert loaders.AutoModelForCausalLM.from_pretrained.call_count == 1

    loaders.peft_model.merge_and_unload.assert_called_once()
    loaders.merged.eval.assert_called_once()
    assert model.model is loaders.merged
    assert loaders.tokenizer.pad_token == loaders.tokenizer.eos_token
    assert loaders.tokenizer.padding_side == "left"
    assert model._eos_ids == [128001, 128009]

    info = model.get_model_info()
    assert info["loaded"] and info["model_type"] == "lora" and info["base_model"] == BASE_ID
    assert info["resolved_path"] == str(lora_env["target"]) and info["version"] == "final-best-model-v9"
    assert info["base_cached"] is True and info["dtype"] == "bfloat16"

    # idempotent: no second load
    assert model.load_model() is True
    assert loaders.AutoModelForCausalLM.from_pretrained.call_count == 1


def test_online_fallback_only_after_local_failure(lora_env, loaders, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HUGGING_FACE_TOKEN", "hf_dummy")
    loaders.AutoModelForCausalLM.from_pretrained.side_effect = [OSError("not in cache"), loaders.base_model]
    model = h.CengBotModel()
    assert model.load_model() is True
    calls = loaders.AutoModelForCausalLM.from_pretrained.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["local_files_only"] is True and calls[0].kwargs["cache_dir"] == str(lora_env["cache"])
    assert "token" not in calls[0].kwargs
    assert calls[1].args == (BASE_ID,)
    assert calls[1].kwargs["local_files_only"] is False
    assert calls[1].kwargs["cache_dir"] == str(lora_env["cache"])
    assert calls[1].kwargs["token"] == "hf_dummy"
    # tokenizer + adapter loaded locally only
    assert all(c.kwargs["local_files_only"] for c in loaders.AutoTokenizer.from_pretrained.call_args_list)
    assert loaders.PeftModel.from_pretrained.call_count == 1


def test_offline_mode_never_tries_online(lora_env, loaders):
    loaders.AutoModelForCausalLM.from_pretrained.side_effect = OSError("not in cache")
    model = h.CengBotModel()
    assert model.load_model() is False
    assert loaders.AutoModelForCausalLM.from_pretrained.call_count == 1
    assert "not available offline" in model.last_load_error
    assert model.model is None and model.tokenizer is None


def test_uncached_base_with_downloads_disabled(tmp_path, monkeypatch, loaders):
    model_dir = make_lora(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(model_dir))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("MODEL_ALLOW_DOWNLOAD", "false")
    loaders.AutoModelForCausalLM.from_pretrained.side_effect = OSError("not in cache")
    model = h.CengBotModel()
    assert model.load_model() is False
    call = loaders.AutoModelForCausalLM.from_pretrained.call_args_list[0]
    assert call.kwargs["local_files_only"] is True and call.kwargs["cache_dir"] == str(tmp_path / "empty-cache")
    assert loaders.AutoModelForCausalLM.from_pretrained.call_count == 1


def test_tokenizer_falls_back_to_cached_base_tokenizer(lora_env, loaders):
    loaders.AutoTokenizer.from_pretrained.side_effect = [OSError("bad tokenizer files"), loaders.tokenizer]
    model = h.CengBotModel()
    assert model.load_model() is True
    calls = loaders.AutoTokenizer.from_pretrained.call_args_list
    assert [c.args[0] for c in calls] == [str(lora_env["target"]), BASE_ID]
    assert all(c.kwargs["local_files_only"] is True for c in calls)
    assert calls[1].kwargs["cache_dir"] == str(lora_env["cache"])


def test_adapter_base_mismatch_is_reported_clearly(lora_env, loaders):
    loaders.PeftModel.from_pretrained.side_effect = RuntimeError("size mismatch for q_proj.lora_A")
    model = h.CengBotModel()
    assert model.load_model() is False
    assert "does not match base model" in model.last_load_error


def test_merge_can_be_disabled(lora_env, loaders, monkeypatch):
    monkeypatch.setenv("MODEL_MERGE_LORA", "false")
    model = h.CengBotModel()
    assert model.load_model()
    loaders.peft_model.merge_and_unload.assert_not_called()
    assert model.model is loaders.peft_model


def test_cpu_fallback_uses_float32_without_device_map(lora_env, loaders, monkeypatch):
    monkeypatch.setattr(h.torch.cuda, "is_available", lambda: False)
    model = h.CengBotModel()
    assert model.load_model()
    call = loaders.AutoModelForCausalLM.from_pretrained.call_args_list[0]
    assert call.kwargs["torch_dtype"] == torch.float32 and "device_map" not in call.kwargs
    loaders.merged.to.assert_called_once_with(torch.device("cpu"))
    assert model.device.type == "cpu"


def test_use_cuda_false_forces_cpu(lora_env, loaders, monkeypatch):
    monkeypatch.setenv("USE_CUDA", "false")
    model = h.CengBotModel()
    assert model.load_model()
    assert loaders.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"] == torch.float32


def test_bf16_unsupported_gpu_uses_fp16(lora_env, loaders, monkeypatch):
    monkeypatch.setattr(h.torch.cuda, "is_bf16_supported", lambda: False)
    model = h.CengBotModel()
    assert model.load_model()
    assert loaders.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"] == torch.float16


def test_full_model_loading(tmp_path, monkeypatch, loaders):
    model_dir = make_full(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(model_dir))
    model = h.CengBotModel()
    assert model.load_model()
    call = loaders.AutoModelForCausalLM.from_pretrained.call_args
    assert call.args == (str(model_dir),) and call.kwargs["local_files_only"] is True
    loaders.PeftModel.from_pretrained.assert_not_called()
    assert model.get_model_info()["model_type"] == "full"


def test_load_failures_log_traceback_once_then_concise(lora_env, loaders, caplog):
    loaders.AutoModelForCausalLM.from_pretrained.side_effect = MemoryError("CUDA out of memory")
    model = h.CengBotModel()
    with caplog.at_level(logging.ERROR, logger=h.logger.name):
        for _ in range(5):
            assert model.load_model() is False
    assert h.handle_model_error.call_count == 1  # full traceback only for the first failure
    records = [r for r in caplog.records if r.name == h.logger.name]
    assert len(records) == 5 and all(r.exc_info is None for r in records)
    assert "failure #5" in records[-1].getMessage()


def test_validation_errors_never_dump_tracebacks(tmp_path, monkeypatch, loaders):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing"))
    model = h.CengBotModel()
    assert model.load_model() is False
    h.handle_model_error.assert_not_called()
    assert "Model directory not found" in model.last_load_error
    assert "Model directory not found" in model.get_model_info()["error"]


def test_reload_model(lora_env, loaders):
    model = h.CengBotModel()
    assert model.load_model()
    assert model.reload_model()
    assert loaders.AutoModelForCausalLM.from_pretrained.call_count == 2
    model.unload_model()
    assert not model.is_loaded()


# --------------------------------------------------------------------------
# Prompt template must match training (src/train_model.py, read-only)
# --------------------------------------------------------------------------

# The templates final-best-model-v1.2 was trained with (src/train_model.py, Config.turkish_template /
# Config.english_template at review time). Inference must reproduce them exactly.
V12_TRAINING_TEMPLATES = {
    "tr": ("Sen Çukurova Üniversitesi Bilgisayar Mühendisliği bölümünün deneyimli dijital asistanısın. "
           "Öğrencilere samimi, yardımsever ve doğru bilgiler vererek destek oluyorsun.\n\n"
           "Öğrenci: {question}\nAsistan: {answer}"),
    "en": ("You are an experienced digital assistant for Cukurova University Computer Engineering Department. "
           "You provide friendly, helpful, and accurate information to support students.\n\n"
           "Student: {question}\nAssistant: {answer}"),
}


@pytest.mark.parametrize("question,lang", [
    ("Staj zorunlu mu?", "tr"),
    ("Is the internship mandatory?", "en"),
])
def test_prompt_matches_training_template(question, lang):
    expected = V12_TRAINING_TEMPLATES[lang].format(question=question, answer="").rstrip()
    model = h.CengBotModel()
    prompt, detected = model.format_prompt(question)
    assert detected == lang
    assert prompt == expected


def test_train_model_fallback_prompt_copies_match_inference():
    """train_model.py (read-only here) keeps fallback copies of the prompt constants."""
    tree = ast.parse((ROOT / "src" / "train_model.py").read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    if "_FALLBACK_SYSTEM_PROMPT_TR" not in values:
        pytest.skip("train_model.py has no fallback prompt copies")
    assert values["_FALLBACK_SYSTEM_PROMPT_TR"] == h.TRAINING_SYSTEM_PROMPT_TR
    assert values["_FALLBACK_SYSTEM_PROMPT_EN"] == h.TRAINING_SYSTEM_PROMPT_EN
    assert values["_FALLBACK_TURN_MARKERS"] == h.TRAINING_TURN_MARKERS
    template = values.get("PROMPT_TEMPLATE")
    if template:
        model = h.CengBotModel()
        for question, lang in (("Staj zorunlu mu?", "tr"), ("Is it mandatory?", "en")):
            user, assistant = h.TRAINING_TURN_MARKERS[lang]
            system = h.TRAINING_SYSTEM_PROMPT_TR if lang == "tr" else h.TRAINING_SYSTEM_PROMPT_EN
            assert template.format(system_prompt=system, user_marker=user, assistant_marker=assistant,
                                   message=question) == model.format_prompt(question, lang)[0]


def test_legacy_prompt_style(monkeypatch):
    monkeypatch.setenv("MODEL_PROMPT_STYLE", "legacy")
    model = h.CengBotModel()
    prompt, lang = model.format_prompt("Staj zorunlu mu?")
    assert lang == "tr"
    assert prompt == f"{h.LEGACY_SYSTEM_PROMPT_TR}\n\nStudent: Staj zorunlu mu?\nAssistant:"


# --------------------------------------------------------------------------
# Generation with a fake char-level tokenizer/model
# --------------------------------------------------------------------------

class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    unk_token_id = None

    def __init__(self):
        self.pad_token = "<eos>"
        self.padding_side = "left"
        self.calls = []

    @property
    def pad_token_id(self):
        return 0 if self.pad_token else None

    def __call__(self, text, return_tensors=None, add_special_tokens=True, split_special_tokens=False, **kw):
        self.calls.append({"text": text, "split_special_tokens": split_special_tokens})
        ids = ([1] if add_special_tokens else []) + [ord(c) for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(chr(i) for i in ids if i > 2)

    def convert_tokens_to_ids(self, token):
        return 2 if token == "<|eot_id|>" else None


class FakeModel:
    """Emits ``continuation`` one char-token at a time, honouring stopping criteria."""

    def __init__(self, continuation, delay=0.0):
        self.continuation = continuation
        self.delay = delay
        self.calls = []
        self.steps = 0
        self.active = 0
        self.max_active = 0
        self.device = torch.device("cpu")
        self._lock = threading.Lock()

    def get_input_embeddings(self):
        raise AttributeError("fake")

    def generate(self, input_ids, attention_mask=None, generation_config=None, stopping_criteria=None):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                threading.Event().wait(self.delay)
            self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask,
                               "generation_config": generation_config})
            ids = input_ids
            for ch in self.continuation[: generation_config.max_new_tokens]:
                ids = torch.cat([ids, torch.tensor([[ord(ch)]])], dim=1)
                self.steps += 1
                if stopping_criteria is not None and bool(stopping_criteria(ids, None).all()):
                    break
            return ids
        finally:
            with self._lock:
                self.active -= 1


def loaded_model(continuation):
    model = h.CengBotModel()
    model.model = FakeModel(continuation)
    model.tokenizer = FakeTokenizer()
    model._eos_ids = [0, 2]
    model.spec = mock.Mock(version="final-best-model-v9")
    return model


def test_generation_decodes_only_new_tokens_and_stops_at_turn_marker():
    continuation = " Evet, staj zorunludur.\nÖğrenci: Teşekkürler!\nAsistan: Rica ederim, başka sorun var mı?"
    model = loaded_model(continuation)
    answer, info = model.generate_response_with_info("Staj zorunlu mu?")
    assert answer == "Evet, staj zorunludur."
    assert "Sen Çukurova" not in answer  # the prompt is never part of the answer
    assert info["stopped_by"] == "turn_marker"
    assert model.model.steps < len(continuation)  # generation stopped early
    assert info["language"] == "tr" and info["model_version"] == "final-best-model-v9"
    assert info["input_tokens"] == model.model.calls[0]["input_ids"].shape[-1]
    cfg = model.model.calls[0]["generation_config"]
    assert cfg.eos_token_id == [0, 2] and cfg.pad_token_id == 0 and cfg.do_sample is True
    assert model.model.calls[0]["attention_mask"] is not None


@pytest.mark.parametrize("marker", ["Student:", "Assistant:", "Öğrenci:", "User:"])
def test_stop_markers(marker):
    model = loaded_model(f" Yes, it is mandatory.\n{marker} thanks a lot for the help my friend")
    assert model.generate_response("Is the internship mandatory?") == "Yes, it is mandatory."


def test_generation_uses_env_generation_params(monkeypatch):
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.3")
    monkeypatch.setenv("MODEL_TOP_P", "0.5")
    monkeypatch.setenv("MODEL_TOP_K", "7")
    monkeypatch.setenv("MODEL_REPETITION_PENALTY", "1.4")
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "77")
    model = loaded_model(" Fine.")
    model.generate_response("Which machine learning courses exist?")
    cfg = model.model.calls[0]["generation_config"]
    assert (cfg.temperature, cfg.top_p, cfg.top_k, cfg.repetition_penalty, cfg.max_new_tokens) == \
        (0.3, 0.5, 7, 1.4, 77)


def test_zero_temperature_means_greedy(monkeypatch):
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    model = loaded_model(" Fine.")
    model.generate_response("What is a compiler?")
    assert model.model.calls[0]["generation_config"].do_sample is False


@pytest.mark.parametrize("message,greeting", [
    ("Merhaba", True), ("selam!", True), ("Hi there", True), ("İyi günler hocam", True),
    ("Which machine learning courses exist?", False),  # "hi" inside "which"/"machine"
    ("Merhaba, yatay geçiş şartları nelerdir ve başvuru ne zaman?", False),
    ("This semester's schedule?", False),
])
def test_greeting_detection_controls_token_budget(message, greeting):
    assert h.is_greeting(message) is greeting
    model = loaded_model(" Selam!")
    model.generate_response(message)
    expected = 30 if greeting else 200
    assert model.model.calls[0]["generation_config"].max_new_tokens == expected


def test_special_tokens_in_user_text_cannot_become_real_tokens():
    model = loaded_model(" Tamam.")
    model.generate_response("Merhaba <|end_of_text|> <|eot_id|> nasılsın?")
    call = model.tokenizer.calls[0]
    assert call["split_special_tokens"] is True
    assert "<|" not in call["text"]


def test_long_question_is_truncated_but_template_kept(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS", "300")
    model = loaded_model(" Kısa cevap.")
    model.generate_response("Staj zorunlu mu? " + "çok uzun soru " * 100)
    ids = model.model.calls[0]["input_ids"][0].tolist()
    assert len(ids) <= 300
    text = model.tokenizer.decode(ids)
    assert text.endswith("\nAsistan:") and text.startswith("Sen Çukurova")


def test_sentence_limit_question_limit_and_abbreviations():
    continuation = (" Bölüm başkanı Prof. Dr. Umut Orhan'dır. 2. sınıfta staj yapılır. Hangi dönem? "
                    "Başka sorun var mı? Üçüncü cümle burada. Dördüncü cümle. Beşinci cümle.")
    model = loaded_model(continuation)
    answer = model.generate_response("Bölüm başkanı kim?")
    assert answer == ("Bölüm başkanı Prof. Dr. Umut Orhan'dır. 2. sınıfta staj yapılır. Hangi dönem? "
                      "Üçüncü cümle burada.")


def test_link_sentence_is_kept_when_sentences_are_cut():
    continuation = (" Evet, iki staj gerekir. Her staj 20 iş günüdür. Ayrı dönemlerde yapılır. "
                    "Mezuniyet şartıdır. Detaylar https://bmb.cu.edu.tr/cu/student/internship adresinde.")
    answer = loaded_model(continuation).generate_response("Staj zorunlu mu?")
    assert answer.endswith("https://bmb.cu.edu.tr/cu/student/internship adresinde.")
    assert answer.count(".") >= 4


def test_sentence_cut_by_token_limit_is_dropped(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "40")
    continuation = " Evet, staj zorunludur. Ayrıca öğrenci işlerine başvurmanız gerekmektedir ve"
    model = loaded_model(continuation)
    answer, info = model.generate_response_with_info("Staj zorunlu mu?")
    assert info["stopped_by"] == "max_tokens"
    assert answer == "Evet, staj zorunludur."


def test_generation_stops_early_once_enough_sentences(monkeypatch):
    """Without a link the model gets max_sentences + 4 closed sentences to produce one."""
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "400")
    sentences = "".join(f" Cümle numarası {i} burada." for i in range(1, 10))  # 25 chars each
    model = loaded_model(sentences)
    answer, info = model.generate_response_with_info("Staj zorunlu mu?")
    assert info["stopped_by"] == "sentence_limit"
    assert model.model.steps == 8 * 25 + 1  # the 8th sentence is closed by the next space
    assert answer == "Cümle numarası 1 burada. Cümle numarası 2 burada. Cümle numarası 3 burada. Cümle numarası 4 burada."


def test_generation_stops_as_soon_as_a_kept_sentence_has_a_link(monkeypatch):
    """A link among the kept sentences makes the answer final: generation used to run on
    for max_sentences + 2 sentences that were all dropped."""
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "400")
    head = " Bilgi https://ebs.cu.edu.tr/Program/555 adresinde. İkinci cümle. Üçüncü cümle. Dördüncü cümle."
    model = loaded_model(head + " Beşinci cümle. Altıncı cümle. Yedinci cümle.")
    answer, info = model.generate_response_with_info("Ders planı nerede?")
    assert info["stopped_by"] == "sentence_limit" and model.model.steps == len(head) + 1
    assert answer == head.strip()


def test_generation_stops_once_the_rescued_link_is_complete(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "400")
    head = (" Evet, iki staj gerekir. Her staj 20 iş günüdür. Ayrı dönemlerde yapılır. Mezuniyet şartıdır."
            " Detaylar https://bmb.cu.edu.tr/staj adresinde.")
    model = loaded_model(head + " Altıncı cümle. Yedinci cümle. Sekizinci cümle.")
    answer, info = model.generate_response_with_info("Staj zorunlu mu?")
    assert info["stopped_by"] == "sentence_limit" and model.model.steps == len(head) + 1
    assert answer == head.strip()


def test_a_link_is_never_cut_at_a_dot_inside_it(monkeypatch):
    """"...ebs.cu." at the end of the text looked like a complete sentence to the old stopper."""
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "400")
    text = (" Bir. İki. Üç. Dört. Bilgi için www.ebs.cu.edu.tr/Program/DersPlan/555/2024 adresine bakın."
            " Altı. Yedi.")
    answer = loaded_model(text).generate_response("Ders planı?")
    assert answer.endswith("www.ebs.cu.edu.tr/Program/DersPlan/555/2024 adresine bakın.")


def _sentence_pool():
    return ["Evet, staj zorunludur.", "Prof. Dr. Ali Hoca dersi verir.", "2. sınıfta alınır.", "Ne zaman?",
            "Başka sorun var mı?", "Detaylar https://ebs.cu.edu.tr/Program/555 adresinde.",
            "Bkz. www.cu.edu.tr sayfası.", "Kayıtlar Eylül'de başlar!", "Ders 3 AKTS'dir…",
            "Yönetmelik (madde 5) geçerlidir.", "Link: https://bmb.cu.edu.tr/staj?", "Başvuru OBS üzerinden yapılır."]


@pytest.mark.parametrize("seed", range(60))
def test_early_stop_never_changes_the_answer(monkeypatch, seed):
    """Property: stopping as soon as answer_is_final() holds gives exactly the answer of the
    whole generation (links only in the first max_sentences + 2 sentences, so the "patience"
    cut never decides)."""
    import random

    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "2000")
    rng = random.Random(seed)
    pool = _sentence_pool()
    count = rng.randint(1, 12)
    chosen = []
    for i in range(count):
        sentence = rng.choice(pool)
        while i >= 6 and h._URL_RE.search(sentence):
            sentence = rng.choice(pool)
        chosen.append(sentence)
    continuation = "".join(rng.choice([" ", " ", "\n"]) + sentence for sentence in chosen)
    early, info = loaded_model(continuation).generate_response_with_info("Staj zorunlu mu?")
    monkeypatch.setattr(h, "answer_is_final", lambda text, limit: False)
    full, full_info = loaded_model(continuation).generate_response_with_info("Staj zorunlu mu?")
    assert full_info["stopped_by"] == "eos"
    assert early == full, (continuation, info["stopped_by"])


def test_sentence_stopper_decodes_only_when_a_token_brings_whitespace():
    class CountingTokenizer(FakeTokenizer):
        def __init__(self):
            super().__init__()
            self.full_decodes = 0

        def decode(self, ids, skip_special_tokens=True):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            if len(ids) > 1:
                self.full_decodes += 1
            return super().decode(ids, skip_special_tokens)

    tokenizer = CountingTokenizer()
    stopper = h.SentenceLimitStoppingCriteria(tokenizer, prompt_length=0, max_sentences=4)
    text = "Uzunkelimelerdenoluşanbircümle. İkinci"
    ids = torch.tensor([[ord(c) for c in text]])
    for n in range(1, len(text) + 1):
        stopper(ids[:, :n], None)
    assert tokenizer.full_decodes == text.count(" ")


def test_sentence_stop_disabled_when_unlimited(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_SENTENCES", "0")
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "120")
    sentences = "".join(f" Cümle {i}." for i in range(1, 40))
    model = loaded_model(sentences)
    _, info = model.generate_response_with_info("Staj zorunlu mu?")
    assert info["stopped_by"] == "max_tokens" and model.model.steps == 120


def test_empty_generation_returns_fallback():
    model = loaded_model("\nÖğrenci: ...")
    assert model.generate_response("Staj zorunlu mu?") == h._FALLBACK_ANSWER["tr"]


def test_generate_is_serialised_by_a_lock():
    model = loaded_model(" Tamam.")
    model.model.delay = 0.05
    threads = [threading.Thread(target=model.generate_response, args=("Staj zorunlu mu?",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(model.model.calls) == 4
    assert model.model.max_active == 1


def test_generation_errors(monkeypatch):
    model = loaded_model(" x")
    model.model.generate = mock.Mock(side_effect=RuntimeError("CUDA error"))
    assert model.generate_response("Staj?") == h._ERROR_ANSWER
    h.handle_model_error.assert_called_once()
    with pytest.raises(RuntimeError, match="CUDA error"):
        model.generate_response_with_info("Staj?", raise_errors=True)


def test_not_loaded():
    model = h.CengBotModel()
    assert model.generate_response("Staj?") == "Model not loaded."
    with pytest.raises(RuntimeError):
        model.generate_response_with_info("Staj?", raise_errors=True)


@pytest.mark.parametrize("text,lang", [
    ("Bilgisayar mühendisliğinde staj zorunlu mu?", "tr"),
    ("yaz okulu var mi", "tr"),
    ("Ödev ne zaman?", "tr"),
    ("Merhaba", "tr"),
    ("Is the internship mandatory?", "en"),
    ("Who is Dr. Şahin?", "en"),
    ("What is Çukurova University?", "en"),
    ("hi", "en"),
])
def test_language_detection_is_deterministic(text, lang):
    model = h.CengBotModel()
    assert {model.detect_language(text) for _ in range(5)} == {lang}


def test_split_sentences_keeps_urls_and_ordinals():
    parts = h.split_sentences("Bkz. https://a.b/c.d sayfası. 3. sınıf! Doç. Dr. Ali? Son")
    assert parts == ["Bkz. https://a.b/c.d sayfası. ", "3. sınıf! ", "Doç. Dr. Ali? ", "Son"]


# --------------------------------------------------------------------------
# Model families: stop tokens, pad token, remote code, reasoning, prompt suffix
# --------------------------------------------------------------------------

class VocabTokenizer:
    """Tokenizer stand-in with a vocabulary, added (special) tokens and an unk token."""

    def __init__(self, vocab, added=(), specials=(), eos=None, unk=None, pad=None):
        self.vocab = dict(vocab)
        self._added = {t: self.vocab[t] for t in added}
        self.all_special_tokens = list(specials)
        self.eos_token = eos
        self.eos_token_id = self.vocab.get(eos) if eos else None
        self.unk_token = unk
        self.unk_token_id = self.vocab.get(unk) if unk else None
        self.pad_token = pad

    def get_added_vocab(self):
        return dict(self._added)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


@pytest.mark.parametrize("family,vocab,added,eos,unk,generation_eos,expected", [
    ("llama3", {"<|end_of_text|>": 128001, "<|eot_id|>": 128009, "</s>": 524}, ["<|end_of_text|>", "<|eot_id|>"],
     "<|end_of_text|>", None, 128001, [128001, 128009]),  # "</s>" is a plain vocabulary piece: not a stop token
    ("qwen2.5-instruct", {"<|endoftext|>": 151643, "<|im_end|>": 151645}, ["<|endoftext|>", "<|im_end|>"],
     "<|im_end|>", None, [151645, 151643], [151645, 151643]),
    ("mistral", {"<unk>": 0, "<s>": 1, "</s>": 2}, ["<unk>", "<s>", "</s>"], "</s>", "<unk>", 2, [2]),
    ("gemma2", {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<end_of_turn>": 107},
     ["<pad>", "<eos>", "<bos>", "<unk>", "<end_of_turn>"], "<eos>", "<unk>", [1, 107], [1, 107]),
    ("phi3", {"<|endoftext|>": 32000, "<|assistant|>": 32001, "<|end|>": 32007, "<|user|>": 32010, "</s>": 2},
     ["<|endoftext|>", "<|assistant|>", "<|end|>", "<|user|>", "</s>"], "<|endoftext|>", None,
     [32000, 32001, 32007], [32000, 32001, 32007, 2, 32010]),
    ("glm4", {"<|endoftext|>": 151329, "<|user|>": 151336, "<|observation|>": 151338},
     ["<|endoftext|>", "<|user|>", "<|observation|>"], "<|endoftext|>", None, [151329, 151336, 151338],
     [151329, 151336, 151338]),
    ("deepseek-r1-distill", {"<｜end▁of▁sentence｜>": 151643, "<think>": 151648},
     ["<｜end▁of▁sentence｜>", "<think>"], "<｜end▁of▁sentence｜>", None, 151643, [151643]),
])
def test_collect_stop_token_ids_per_family(family, vocab, added, eos, unk, generation_eos, expected):
    tok = VocabTokenizer(vocab, added=added, eos=eos, unk=unk)
    model = mock.Mock(generation_config=mock.Mock(eos_token_id=generation_eos))
    assert h.collect_stop_token_ids(tok, model) == expected
    assert h.CengBotModel._collect_eos_ids(tok, model) == expected


def test_collect_stop_token_ids_edge_cases():
    # unknown candidates map to unk -> ignored; bools / negatives / junk in generation_config ignored
    tok = VocabTokenizer({"<unk>": 0, "</s>": 2}, added=["<unk>"], specials=["</s>"], eos="</s>", unk="<unk>")
    gen = mock.Mock(generation_config=mock.Mock(eos_token_id=[True, -1, "x", None, [2, 5]]))
    assert h.collect_stop_token_ids(tok, gen) == [2, 5]
    assert h.collect_stop_token_ids(tok, mock.Mock(generation_config=None)) == [2]

    class Minimal:  # no get_added_vocab / all_special_tokens: any real id that is not unk counts
        eos_token_id = 7
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 9, "<eos>": 0}.get(token)

    assert h.collect_stop_token_ids(Minimal()) == [7, 9]


def test_ensure_pad_token_is_generic(caplog):
    tok = mock.Mock(pad_token=None, eos_token="<|im_end|>")
    h.ensure_pad_token(tok)
    assert tok.pad_token == "<|im_end|>"
    tok = mock.Mock(pad_token="<pad>", eos_token="<eos>")  # Gemma: keeps its own pad token
    h.ensure_pad_token(tok)
    assert tok.pad_token == "<pad>"
    tok = mock.Mock(pad_token=None, eos_token=None, unk_token="<unk>")
    h.ensure_pad_token(tok)
    assert tok.pad_token == "<unk>"
    tok = mock.Mock(pad_token=None, eos_token=None, unk_token=None)
    with caplog.at_level(logging.WARNING, logger="llama_model_handler"):
        h.ensure_pad_token(tok)
    assert "neither a pad nor an EOS token" in caplog.text


def test_load_model_uses_generation_config_stop_tokens_and_reports_architecture(lora_env, loaders):
    loaders.merged.generation_config.eos_token_id = [128001, 128008, 128009]
    loaders.merged.config.model_type = "llama"
    model = h.CengBotModel()
    assert model.load_model()
    assert model._eos_ids == [128001, 128008, 128009]
    info = model.get_model_info()
    assert info["architecture"] == "llama" and info["stop_token_ids"] == [128001, 128008, 128009]
    assert info["trust_remote_code"] is False


def test_trust_remote_code_is_opt_in_and_warned(lora_env, loaders, monkeypatch, caplog):
    monkeypatch.setenv("MODEL_TRUST_REMOTE_CODE", "true")
    model = h.CengBotModel()
    with caplog.at_level(logging.WARNING, logger="llama_model_handler"):
        assert model.load_model()
    tok_call = loaders.AutoTokenizer.from_pretrained.call_args_list[0]
    base_call = loaders.AutoModelForCausalLM.from_pretrained.call_args_list[0]
    assert tok_call.kwargs["trust_remote_code"] is True and base_call.kwargs["trust_remote_code"] is True
    assert tok_call.kwargs["local_files_only"] is True  # still offline-first
    assert "SECURITY" in caplog.text and "MODEL_TRUST_REMOTE_CODE" in caplog.text
    assert model.get_model_info()["trust_remote_code"] is True


def test_4bit_inference_is_opt_in_and_keeps_the_adapter_unmerged(lora_env, loaders, monkeypatch):
    model = h.CengBotModel()
    assert model.load_model()
    assert "quantization_config" not in loaders.AutoModelForCausalLM.from_pretrained.call_args.kwargs
    assert model.get_model_info()["load_in_4bit"] is False

    monkeypatch.setenv("MODEL_LOAD_IN_4BIT", "true")
    assert model.reload_model()
    quant = loaders.AutoModelForCausalLM.from_pretrained.call_args.kwargs["quantization_config"]
    assert quant.load_in_4bit and quant.bnb_4bit_quant_type == "nf4" and quant.bnb_4bit_compute_dtype == torch.bfloat16
    assert loaders.peft_model.merge_and_unload.call_count == 1  # only the first (bf16) load merged
    assert model.model is loaders.peft_model
    assert model.get_model_info()["load_in_4bit"] is True


def test_4bit_inference_needs_cuda(lora_env, loaders, monkeypatch, caplog):
    monkeypatch.setenv("MODEL_LOAD_IN_4BIT", "true")
    monkeypatch.setenv("USE_CUDA", "false")
    model = h.CengBotModel()
    with caplog.at_level(logging.WARNING, logger="llama_model_handler"):
        assert model.load_model()
    assert "quantization_config" not in loaders.AutoModelForCausalLM.from_pretrained.call_args.kwargs
    assert "MODEL_LOAD_IN_4BIT needs a CUDA GPU" in caplog.text
    assert model.get_model_info()["load_in_4bit"] is False


@pytest.mark.parametrize("message,needle", [
    ("The repository THUDM/chatglm3-6b contains custom code which must be executed to correctly load the model. "
     "Please pass the argument `trust_remote_code=True` to allow custom code to be run.", "MODEL_TRUST_REMOTE_CODE=true"),
    ("The checkpoint you are trying to load has model type `newarch` but Transformers does not recognize this "
     "architecture.", "does not support the architecture"),
])
def test_unsupported_architectures_fail_with_a_clear_error(lora_env, loaders, message, needle):
    loaders.AutoModelForCausalLM.from_pretrained.side_effect = ValueError(message)
    model = h.CengBotModel()
    assert model.load_model() is False
    assert model.last_load_error.startswith("ModelLoadError") and needle in model.last_load_error
    h.handle_model_error.assert_not_called()  # a clear message, no traceback dump


def test_reasoning_blocks_are_not_part_of_the_answer():
    model = loaded_model("<think>Kullanıcı staj soruyor. Kısa cevap vermeliyim.</think> Evet, staj zorunludur.")
    assert model.generate_response("Staj zorunlu mu?") == "Evet, staj zorunludur."
    model = loaded_model("<think>Uzun bir düşünce, hiç bitmeyen")
    assert model.generate_response("Staj zorunlu mu?") == h._FALLBACK_ANSWER["tr"]
    model = loaded_model(" Evet.<|im_end|><|im_start|>user teşekkürler")
    assert model.generate_response("Staj zorunlu mu?") == "Evet."


def test_sentence_stopper_ignores_reasoning(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "400")
    reasoning = "".join(f" Düşünce {i} burada." for i in range(1, 15))
    continuation = f"<think>{reasoning}</think> Evet, staj zorunludur."
    model = loaded_model(continuation)
    answer, info = model.generate_response_with_info("Staj zorunlu mu?")
    assert answer == "Evet, staj zorunludur."
    assert model.model.steps == len(continuation)  # never stopped inside the reasoning
    assert info["stopped_by"] == "eos"


class EosAppendingTokenizer(FakeTokenizer):
    """Appends EOS (id 0) after the text when add_special_tokens=True (e.g. some GPT-style tokenizers)."""

    def __call__(self, text, return_tensors=None, add_special_tokens=True, split_special_tokens=False, **kw):
        self.calls.append({"text": text, "split_special_tokens": split_special_tokens})
        ids = ([1] if add_special_tokens else []) + [ord(c) for c in text] + ([0] if add_special_tokens else [])
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids}


def test_appended_special_tokens_are_removed_from_the_prompt():
    assert h.count_appended_special_tokens(FakeTokenizer()) == 0
    assert h.count_appended_special_tokens(EosAppendingTokenizer()) == 1
    assert h.count_appended_special_tokens(mock.MagicMock()) == 0
    model = loaded_model(" Evet.")
    model.tokenizer = EosAppendingTokenizer()
    model._prompt_suffix_tokens = h.count_appended_special_tokens(model.tokenizer)
    assert model.generate_response("Staj zorunlu mu?") == "Evet."
    call = model.model.calls[0]
    ids = call["input_ids"][0].tolist()
    assert ids[0] == 1 and ids[-1] != 0  # BOS kept, appended EOS removed
    assert model.tokenizer.decode(ids).endswith("\nAsistan:")
    assert call["attention_mask"].shape == call["input_ids"].shape


# --------------------------------------------------------------------------
# Real GPU smoke test (opt-in)
# --------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1 to load the real model on the GPU")
def test_gpu_smoke_real_model_offline(monkeypatch, capsys):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    # real model + real default HF cache lookup
    monkeypatch.setattr(h, "_default_hf_cache", _REAL_DEFAULT_HF_CACHE)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")  # deterministic

    # Prove offline loading: any non-local network access fails the test.
    attempts = []
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def _is_local(host):
        return host in (None, "", "localhost", "127.0.0.1", "::1") or str(host).startswith("/")

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if not _is_local(host):
            attempts.append(address)
            raise OSError(f"network disabled in GPU smoke test: {address}")
        return real_connect(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_local(host):
            attempts.append(host)
            raise socket.gaierror(f"DNS disabled in GPU smoke test: {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    model = h.CengBotModel()
    started = time.monotonic()
    assert model.load_model(), model.last_load_error
    load_time = time.monotonic() - started
    try:
        info = model.get_model_info()
        assert info["model_type"] == "lora" and info["base_cached"] is True
        assert attempts == [], f"network access attempted: {attempts}"

        results = []
        for question in ("Bilgisayar mühendisliğinde staj zorunlu mu?",
                         "Is the internship mandatory?"):
            t0 = time.monotonic()
            answer, gen_info = model.generate_response_with_info(question, raise_errors=True)
            results.append((question, answer, gen_info, time.monotonic() - t0))
            assert answer.strip()
            assert h.prompt_guard.find_turn_marker(answer) is None
            assert "<|" not in answer
        assert results[0][2]["language"] == "tr" and results[1][2]["language"] == "en"
        with capsys.disabled():
            print(f"\n[gpu-smoke] model={info['version']} load_time={load_time:.1f}s device={info['device']} "
                  f"dtype={info['dtype']}")
            for question, answer, gen_info, seconds in results:
                print(f"[gpu-smoke] Q: {question}\n[gpu-smoke] A ({seconds:.1f}s, "
                      f"{gen_info['output_tokens']} tok, {gen_info['stopped_by']}): {answer}")
    finally:
        model.unload_model()

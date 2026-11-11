"""
Model-family support, end to end: training, loading, validation and switching
work for the common open-weight architectures, not just Llama.

For every family a tiny randomly initialised model (2 layers, hidden size 64)
is built from its transformers config class and saved as a local base model,
together with a tiny byte-level tokenizer that carries the family's special
tokens (BOS or no BOS, its EOS / end-of-turn tokens). The tokenizers only
emulate the families' special-token layout; they are not the real vocabularies.

Per family: ``train_model.main()`` trains a few CPU steps with
``--lora-target-modules auto``, the output satisfies the model-directory
contract, ``scripts/switch_model.sh validate`` accepts it, and the inference
handler loads it, collects the right stop tokens, builds the same prompt
tokens as training and generates text.

Everything runs offline on the CPU. The QLoRA test at the end needs the GPU
and the cached Llama 3.2 3B (marked ``gpu``, skipped unless RUN_GPU_TESTS=1).
"""

import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import transformers

import llama_model_handler as h
import train_model as tm

ROOT = Path(__file__).resolve().parent.parent
SWITCH = ROOT / "scripts" / "switch_model.sh"
MODEL_CACHE = ROOT / "model_cache"
LLAMA = "meta-llama/Llama-3.2-3B"

# Environment that would change training / loading behaviour (the real .env may
# have put some of them into os.environ).
_MODEL_ENV = (
    "BASE_MODEL_NAME", "MODEL_PATH", "LORA_MODEL_PATH", "MODEL_CACHE_DIR", "MODEL_TEMPERATURE",
    "MODEL_MAX_NEW_TOKENS", "MODEL_PRECISION", "USE_CUDA", "MODEL_MAX_INPUT_TOKENS", "MODEL_MAX_SENTENCES",
    "MODEL_PROMPT_STYLE", "MODEL_MERGE_LORA", "MODEL_ALLOW_DOWNLOAD", "MODEL_TRUST_REMOTE_CODE",
    "LORA_TARGET_MODULES", "TRAINING_VERSION", "TRAINING_DATASET",
)


# ---------------------------------------------------------------------------
# Tiny tokenizers with family-specific special tokens
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Flavor:
    specials: Tuple[str, ...]
    eos: str
    stops: Tuple[str, ...]                 # stop tokens the handler must collect
    bos: Optional[str] = None
    pad: Optional[str] = None
    unk: Optional[str] = None
    prefix: Tuple[str, ...] = ()           # tokens the tokenizer puts before the text
    suffix: Tuple[str, ...] = ()           # tokens the tokenizer appends after the text
    generation_eos: Tuple[str, ...] = ()   # model.generation_config.eos_token_id (list)


DS_BOS, DS_EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"

FLAVORS: Dict[str, Flavor] = {
    # Llama 3 / Falcon3 / SmolLM style: BOS, no pad token
    "llama3": Flavor(("<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", "<|start_header_id|>",
                      "<|end_header_id|>"), eos="<|end_of_text|>", bos="<|begin_of_text|>",
                     prefix=("<|begin_of_text|>",), stops=("<|end_of_text|>", "<|eot_id|>")),
    # Qwen (ChatML): NO BOS, pad == eos
    "chatml": Flavor(("<|endoftext|>", "<|im_start|>", "<|im_end|>"), eos="<|endoftext|>", pad="<|endoftext|>",
                     stops=("<|endoftext|>", "<|im_end|>")),
    # DeepSeek-R1 distills: full-width-bar BOS/EOS
    "deepseek": Flavor((DS_BOS, DS_EOS, "<｜User｜>", "<｜Assistant｜>"), eos=DS_EOS, bos=DS_BOS, prefix=(DS_BOS,),
                       stops=(DS_EOS,)),
    # Mistral / Llama 2 (SentencePiece): <s> BOS, </s> EOS, an unk token
    "sentencepiece": Flavor(("<unk>", "<s>", "</s>"), eos="</s>", bos="<s>", unk="<unk>", prefix=("<s>",),
                            stops=("</s>",)),
    # Gemma: BOS required, own pad token, <end_of_turn>
    "gemma": Flavor(("<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"), eos="<eos>",
                    bos="<bos>", pad="<pad>", unk="<unk>", prefix=("<bos>",), stops=("<eos>", "<end_of_turn>")),
    # Phi-3: <|end|> / <|user|>, generation_config.eos_token_id is a list
    "phi3": Flavor(("<unk>", "<s>", "</s>", "<|endoftext|>", "<|assistant|>", "<|end|>", "<|user|>"),
                   eos="<|endoftext|>", bos="<s>", pad="<|endoftext|>", unk="<unk>",
                   stops=("<|endoftext|>", "</s>", "<|end|>", "<|user|>", "<|assistant|>"),
                   generation_eos=("<|endoftext|>", "<|assistant|>", "<|end|>")),
    # GLM-4: [gMASK]<sop> prefix, <|user|> / <|observation|> end a turn
    "glm": Flavor(("<|endoftext|>", "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>",
                   "<|observation|>"), eos="<|endoftext|>", pad="<|endoftext|>", prefix=("[gMASK]", "<sop>"),
                  stops=("<|endoftext|>", "<|user|>", "<|observation|>")),
    # Granite 3
    "granite": Flavor(("<|end_of_text|>", "<|start_of_role|>", "<|end_of_role|>"), eos="<|end_of_text|>",
                      pad="<|end_of_text|>", stops=("<|end_of_text|>",)),
    # GPT-style (OLMo, old Falcon): one <|endoftext|>, separate pad
    "gpt": Flavor(("<|endoftext|>", "<|pad|>"), eos="<|endoftext|>", pad="<|pad|>", stops=("<|endoftext|>",)),
    # A tokenizer that APPENDS its EOS (post-processor "$A <eos>")
    "eos_appending": Flavor(("<|endoftext|>", "<|pad|>"), eos="<|endoftext|>", pad="<|pad|>",
                            suffix=("<|endoftext|>",), stops=("<|endoftext|>",)),
}


def build_tokenizer(directory: Path, flavor: Flavor):
    """Byte-level BPE without merges (256 byte symbols) + the flavor's special tokens."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate(flavor.specials)}
    for ch in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab.setdefault(ch, len(vocab))
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=flavor.unk))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    raw.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in flavor.specials])
    if flavor.prefix or flavor.suffix:
        single = " ".join([*flavor.prefix, "$A", *flavor.suffix])
        raw.post_processor = processors.TemplateProcessing(
            single=single, pair=f"{single} $B",
            special_tokens=[(t, vocab[t]) for t in dict.fromkeys([*flavor.prefix, *flavor.suffix])],
        )
    fast = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token=flavor.bos, eos_token=flavor.eos,
                                   pad_token=flavor.pad, unk_token=flavor.unk)
    fast.save_pretrained(str(directory))
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True)


# ---------------------------------------------------------------------------
# Tiny models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Family:
    name: str
    config_class: str
    model_type: str
    flavor: str
    lora_mode: str                          # what --lora-target-modules auto must pick
    fused: Tuple[str, ...] = field(default=())  # Linear names that only all-linear reaches


FAMILIES = [
    Family("llama", "LlamaConfig", "llama", "llama3", "explicit"),
    Family("qwen2", "Qwen2Config", "qwen2", "chatml", "explicit"),
    Family("deepseek-r1-distill-qwen", "Qwen2Config", "qwen2", "deepseek", "explicit"),
    Family("qwen3", "Qwen3Config", "qwen3", "chatml", "explicit"),
    Family("mistral", "MistralConfig", "mistral", "sentencepiece", "explicit"),
    Family("gemma2", "Gemma2Config", "gemma2", "gemma", "explicit"),
    Family("phi3", "Phi3Config", "phi3", "phi3", "all-linear", fused=("qkv_proj", "gate_up_proj")),
    Family("glm", "GlmConfig", "glm", "glm", "all-linear", fused=("gate_up_proj",)),
    Family("glm4", "Glm4Config", "glm4", "glm", "all-linear", fused=("gate_up_proj",)),
    Family("granite", "GraniteConfig", "granite", "granite", "explicit"),
    Family("olmo2", "Olmo2Config", "olmo2", "gpt", "explicit"),
    Family("falcon", "FalconConfig", "falcon", "gpt", "all-linear", fused=("query_key_value",)),
]


def tiny_config(config_class: str, vocab_size: int, tokenizer):
    cls = getattr(transformers, config_class)
    kwargs = dict(
        vocab_size=vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=1024, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    if config_class != "FalconConfig":
        kwargs.update(intermediate_size=128, num_key_value_heads=2, head_dim=16)
    return cls(**kwargs)


def build_base_model(directory: Path, family: Family):
    """Save a tiny random ``family`` model + tokenizer as a local base model directory."""
    flavor = FLAVORS[family.flavor]
    tokenizer = build_tokenizer(directory, flavor)
    vocab_size = (len(tokenizer) + 63) // 64 * 64
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(tiny_config(family.config_class, vocab_size, tokenizer))
    if flavor.generation_eos:
        model.generation_config.eos_token_id = [tokenizer.convert_tokens_to_ids(t) for t in flavor.generation_eos]
    model.save_pretrained(str(directory), safe_serialization=True)
    return tokenizer


def training_rows():
    rows = []
    for i in range(4):
        rows.append({"question": f"Staj {i}. dönemde zorunlu mu?",
                     "answer": f"Evet, {i}. dönemde staj zorunludur. Ayrıntılar bölüm sayfasında.",
                     "language": "turkish"})
        rows.append({"question": f"Is course {i} mandatory?",
                     "answer": f"Yes, course {i} is mandatory. See the department page.", "language": "english"})
    return rows


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def run_training(argv: List[str]) -> int:
    """train_model.main() without the project .env and without leaking its logger/config."""
    saved_config = tm.config
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tm, "_load_env", lambda: None)
        for name in _MODEL_ENV:
            mp.delenv(name, raising=False)
        try:
            return tm.main(argv)
        finally:
            tm.config = saved_config
            for handler in list(tm.logger.handlers):
                tm.logger.removeHandler(handler)
                handler.close()


def run_switch(*args, root: Path, **env) -> subprocess.CompletedProcess:
    full_env = dict(os.environ, CENGBOT_ROOT=str(root), CENGBOT_PYTHON=sys.executable, CENGBOT_NO_RESTART="1",
                    NO_COLOR="1")
    for name in ("MODEL_PATH", "ASSUME_YES", *_MODEL_ENV):
        full_env.pop(name, None)
    full_env.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", str(SWITCH), *map(str, args)], cwd=str(root), env=full_env,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)


# ---------------------------------------------------------------------------
# One training run per family (module scope: each family is trained once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", params=FAMILIES, ids=[f.name for f in FAMILIES])
def trained(request, tmp_path_factory):
    family: Family = request.param
    if not hasattr(transformers, family.config_class):
        pytest.skip(f"transformers {transformers.__version__} has no {family.config_class}")
    root = tmp_path_factory.mktemp(family.name)
    base_dir = root / f"base-{family.name}"
    tokenizer = build_base_model(base_dir, family)
    dataset = write_jsonl(root / "train.jsonl", training_rows())
    name = f"{family.name}-v1"
    argv = [
        "--base-model", str(base_dir), "--dataset", str(dataset), "--models-dir", str(root / "models"),
        "--checkpoint-dir", str(root / "ckpt"), "--log-dir", str(root / "logs"),
        "--cache-dir", str(root / "empty-hf-cache"), "--output-name", name, "--no-db-export", "--cpu",
        "--epochs", "1", "--batch-size", "4", "--grad-accum", "1", "--val-ratio", "0",
        "--lora-r", "4", "--lora-alpha", "8", "--lora-target-modules", "auto",
    ]
    started = time.monotonic()
    assert run_training(argv) == 0
    final_dir = root / "models" / name
    return SimpleNamespace(
        family=family, flavor=FLAVORS[family.flavor], root=root, base_dir=base_dir, final_dir=final_dir,
        tokenizer=tokenizer, info=json.loads((final_dir / "training_info.json").read_text(encoding="utf-8")),
        adapter_cfg=json.loads((final_dir / "adapter_config.json").read_text(encoding="utf-8")),
        train_seconds=time.monotonic() - started,
    )


def test_auto_lora_targets_match_the_architecture(trained):
    info, family = trained.info, trained.family
    assert info["architecture"] == family.model_type
    assert info["lora_target_modules_requested"] == "auto"
    assert info["lora_target_modules_mode"] == family.lora_mode
    targets = set(info["lora_target_modules"])
    adapter_targets = {t.rsplit(".", 1)[-1] for t in trained.adapter_cfg["target_modules"]}
    assert adapter_targets == targets
    if family.lora_mode == "explicit":
        assert targets == set(tm.LORA_TARGET_MODULES)
    else:
        # fused projections are adapted, the output head never is
        assert set(family.fused) <= targets
        assert "lm_head" not in targets and "embed_tokens" not in targets
    assert info["total_steps"] >= 1 and info["final_loss"] is not None


def test_output_dir_follows_the_model_contract(trained):
    files = {p.name for p in trained.final_dir.iterdir()}
    assert {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "training_info.json"} <= files
    assert not files & {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin",
                        "method1"}
    assert trained.adapter_cfg["base_model_name_or_path"] == str(trained.base_dir)
    assert trained.adapter_cfg["peft_type"] == "LORA" and trained.adapter_cfg["task_type"] == "CAUSAL_LM"
    info = trained.info
    # eager attention only where transformers recommends it for training
    if trained.family.model_type == "gemma2":
        assert info["attn_implementation"] == "eager"
    else:
        assert info["attn_implementation"] != "eager"
    tok_info = info["tokenizer"]
    assert tok_info["adds_bos"] is bool(trained.flavor.bos and trained.flavor.bos in trained.flavor.prefix)
    assert tok_info["eos_token"] == trained.flavor.eos
    assert tok_info["pad_token"] == (trained.flavor.pad or trained.flavor.eos)  # pad := eos when missing
    assert info["trust_remote_code"] is False


def test_switch_model_validate_accepts_the_trained_model(trained, tmp_path):
    project = tmp_path / "proj"
    (project / "models").mkdir(parents=True)
    proc = run_switch("validate", trained.final_dir, root=project)
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "valid - LoRA adapter" in proc.stdout
    assert f"(architecture: {trained.family.model_type})" in proc.stdout
    assert "[WARNING]" not in output and "[ERROR]" not in output


def test_handler_loads_the_model_and_generates(trained, monkeypatch):
    for name in _MODEL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_PATH", str(trained.final_dir))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(trained.root / "empty-hf-cache"))
    monkeypatch.setenv("USE_CUDA", "false")
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "8")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    spec = h.resolve_model_spec(h.ModelConfig())
    assert spec.kind == "lora" and spec.base_model == str(trained.base_dir) and spec.base.cached
    assert spec.tokenizer_source == str(trained.final_dir)

    model = h.CengBotModel()
    assert model.load_model(), model.last_load_error
    try:
        tok = model.tokenizer
        # stop tokens: tokenizer EOS + generation_config + the family's end-of-turn tokens
        expected = {tok.convert_tokens_to_ids(t) for t in trained.flavor.stops}
        assert set(model._eos_ids) == expected
        assert model._eos_ids[0] == tok.eos_token_id
        assert tok.pad_token_id is not None and tok.padding_side == "left"

        # the inference prompt is token-for-token the training prompt (BOS / no BOS / prefix)
        question = "Bilgisayar mühendisliğinde staj zorunlu mu?"
        features, status = tm.encode_example(tok, question, "Evet.", "turkish", 512)
        assert status == "ok"
        prompt_ids = model._encode(question, "tr")["input_ids"][0].tolist()
        assert features["input_ids"][:len(prompt_ids)] == prompt_ids
        assert features["labels"][:len(prompt_ids)] == [tm.IGNORE_INDEX] * len(prompt_ids)
        if trained.flavor.prefix:
            assert prompt_ids[0] == tok.convert_tokens_to_ids(trained.flavor.prefix[0])
        else:
            assert prompt_ids[0] not in set(tok.all_special_ids)

        answer, gen_info = model.generate_response_with_info(question, raise_errors=True, language="tr")
        assert isinstance(answer, str) and answer.strip()
        assert gen_info["output_tokens"] >= 1 and gen_info["stopped_by"] in ("eos", "max_tokens", "turn_marker",
                                                                             "sentence_limit")
        info = model.get_model_info()
        assert info["architecture"] == trained.family.model_type
        assert info["base_model"] == str(trained.base_dir) and info["stop_token_ids"] == model._eos_ids
    finally:
        model.unload_model()


# ---------------------------------------------------------------------------
# Tokenizers that append special tokens, real Llama tokenizer
# ---------------------------------------------------------------------------
def test_eos_appending_tokenizer_never_ends_the_prompt_with_eos(tmp_path):
    tok = build_tokenizer(tmp_path / "tok", FLAVORS["eos_appending"])
    eos = tok.eos_token_id
    assert tok("Merhaba")["input_ids"][-1] == eos  # the tokenizer appends EOS ...
    assert tm._appended_special_token_count(tok) == h.count_appended_special_tokens(tok) == 1

    question, answer = "Staj zorunlu mu?", "Evet, zorunludur."
    features, status = tm.encode_example(tok, question, answer, "turkish", 512)
    assert status == "ok"
    handler = h.CengBotModel()
    handler.tokenizer, handler._prompt_suffix_tokens = tok, h.count_appended_special_tokens(tok)
    encoded = handler._encode(question, "tr")
    prompt_ids = encoded["input_ids"][0].tolist()
    assert encoded["attention_mask"].shape == encoded["input_ids"].shape
    assert prompt_ids[-1] != eos  # ... but never in the middle of a training example / at the end of a prompt
    assert features["input_ids"][:len(prompt_ids)] == prompt_ids
    assert features["input_ids"].count(eos) == 1 and features["labels"][-1] == eos
    assert tm.check_tokenization_consistency(tok, [{"question": question, "answer": answer, "language": "turkish"}]) == 0


@pytest.mark.skipif(not tm.is_model_cached(LLAMA, str(MODEL_CACHE)), reason="Llama 3.2 tokenizer not in model_cache/")
def test_real_llama_tokenizer_keeps_the_production_stop_tokens():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LLAMA, cache_dir=str(MODEL_CACHE), local_files_only=True)
    assert h.collect_stop_token_ids(tok) == [128001, 128009]  # <|end_of_text|>, <|eot_id|> - nothing else
    assert h.count_appended_special_tokens(tok) == 0
    gen = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[128001, 128008, 128009]))
    assert h.collect_stop_token_ids(tok, gen) == [128001, 128008, 128009]


# ---------------------------------------------------------------------------
# Display strings name the active model, not a hardcoded one
# ---------------------------------------------------------------------------
def test_legacy_bots_show_the_active_base_model(tmp_path, monkeypatch):
    import telegram_bot as tb
    import telegram_bot_standalone as sb

    adapter = tmp_path / "models" / "qwen-v1"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct"}))
    legacy = tmp_path / "models" / "legacy" / "method1"
    legacy.mkdir(parents=True)
    (legacy / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "mistralai/Mistral-7B-v0.3"}))
    monkeypatch.setattr(sb, "model_instance", None)

    monkeypatch.setenv("MODEL_PATH", str(adapter))
    assert tb.model_display_name() == sb.model_display_name() == "Qwen2.5-7B-Instruct + LoRA"
    monkeypatch.setenv("MODEL_PATH", str(legacy.parent))
    assert tb.model_display_name() == "Mistral-7B-v0.3 + LoRA"
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing"))
    assert tb.model_display_name() == sb.model_display_name() == "fine-tuned language model"

    loaded = SimpleNamespace(get_model_info=lambda: {"base_model": "google/gemma-2-9b-it"})
    monkeypatch.setattr(sb, "model_instance", loaded)
    assert sb.model_display_name() == "gemma-2-9b-it + LoRA"

    for module in ("telegram_bot.py", "telegram_bot_standalone.py", "data_augmentation.py"):
        source = (ROOT / "src" / module).read_text(encoding="utf-8")
        assert "LLaMA 3.2" not in source and "Llama 3.2 3B" not in source, module


@pytest.mark.parametrize("module", ["telegram_bot", "telegram_bot_standalone"])
async def test_start_and_stats_texts_use_the_model_name(module, tmp_path, monkeypatch, db_session):
    from unittest.mock import AsyncMock, MagicMock

    mod = __import__(module)
    adapter = tmp_path / "phi-v1"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "microsoft/Phi-4-mini-instruct"}))
    monkeypatch.setenv("MODEL_PATH", str(adapter))
    if hasattr(mod, "model_instance"):
        monkeypatch.setattr(mod, "model_instance", None)
    monkeypatch.setattr(mod, "is_chat_allowed", lambda chat_id: True)
    message = MagicMock()
    message.reply_text = AsyncMock()
    update = SimpleNamespace(effective_message=message, effective_user=None)
    await mod.start_command(update, None)
    await mod.stats_command(update, None)
    texts = [call.args[0] for call in message.reply_text.await_args_list]
    assert "<b>AI Engine:</b> Phi-4-mini-instruct + LoRA" in texts[0]
    assert "<b>Model:</b> Phi-4-mini-instruct + LoRA" in texts[1]


def test_augmentation_prompts_do_not_name_a_base_model():
    source = (ROOT / "src" / "data_augmentation.py").read_text(encoding="utf-8")
    assert "küçük bir dil modelini fine-tune etmek" in source
    assert "fine-tune a small language model" in source


# ---------------------------------------------------------------------------
# GPU: QLoRA (4-bit) training smoke test on the real cached base model
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="set RUN_GPU_TESTS=1 to train on the GPU")
def test_gpu_qlora_smoke_train_and_load_llama_3b(tmp_path, monkeypatch, capsys):
    """2 QLoRA steps on Llama 3.2 3B (the path 7-12B models need on a 24 GB GPU), then
    load the adapter on a bf16 base and on a 4-bit base (MODEL_LOAD_IN_4BIT), offline,
    and generate once each. Writes only to tmp_path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not tm.is_model_cached(LLAMA, str(MODEL_CACHE)):
        pytest.skip("meta-llama/Llama-3.2-3B is not cached in model_cache/")
    models_dir = tmp_path / "models"
    assert ROOT / "models" not in (models_dir, *models_dir.parents)
    rows = training_rows()
    rows += [{"question": f"Ders {i} kaç kredi?", "answer": f"Ders {i} 4 kredidir.", "language": "turkish"}
             for i in range(6)]
    rows += [{"question": f"How many credits is course {i}?", "answer": f"Course {i} is 4 credits.",
              "language": "english"} for i in range(6)]
    assert len(rows) == 20
    dataset = write_jsonl(tmp_path / "tiny.jsonl", rows)
    argv = [
        "--base-model", LLAMA, "--cache-dir", str(MODEL_CACHE), "--dataset", str(dataset),
        "--models-dir", str(models_dir), "--checkpoint-dir", str(tmp_path / "ckpt"), "--log-dir", str(tmp_path / "logs"),
        "--output-name", "qlora-smoke", "--no-db-export", "--load-in-4bit", "--lora-target-modules", "auto",
        "--epochs", "1", "--batch-size", "4", "--grad-accum", "4", "--val-ratio", "0", "--workers", "0",
    ]
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    assert run_training(argv) == 0
    train_seconds = time.monotonic() - started
    train_peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
    final_dir = models_dir / "qlora-smoke"
    info = json.loads((final_dir / "training_info.json").read_text(encoding="utf-8"))
    assert info["load_in_4bit"] is True and info["total_steps"] == 2
    assert info["lora_target_modules_mode"] == "explicit" and info["architecture"] == "llama"
    assert json.loads((final_dir / "adapter_config.json").read_text())["base_model_name_or_path"] == LLAMA
    assert not (ROOT / "models" / "qlora-smoke").exists()
    gc.collect()
    torch.cuda.empty_cache()

    for name in _MODEL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_PATH", str(final_dir))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(MODEL_CACHE))
    monkeypatch.setenv("MODEL_TEMPERATURE", "0")
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "40")
    with capsys.disabled():
        print(f"\n[gpu-qlora] train {train_seconds:.1f}s ({info['total_steps']} steps, "
              f"peak {train_peak_gib:.2f} GiB allocated)")
    for load_in_4bit in (False, True):
        monkeypatch.setenv("MODEL_LOAD_IN_4BIT", str(load_in_4bit).lower())
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        model = h.CengBotModel()
        assert model.load_model(), model.last_load_error
        try:
            load_seconds = time.monotonic() - started
            model_info = model.get_model_info()
            assert model_info["base_cached"] is True and model_info["load_in_4bit"] is load_in_4bit
            assert model._eos_ids[:2] == [128001, 128009]
            answer, gen_info = model.generate_response_with_info("Bilgisayar mühendisliğinde staj zorunlu mu?",
                                                                 raise_errors=True)
            assert answer.strip()
            infer_peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
            with capsys.disabled():
                print(f"[gpu-qlora] inference base={'4-bit' if load_in_4bit else 'bf16'}: load {load_seconds:.1f}s, "
                      f"generate {gen_info['generation_time_s']}s ({gen_info['output_tokens']} tok), peak "
                      f"{infer_peak_gib:.2f} GiB | answer: {answer[:100]!r}")
        finally:
            model.unload_model()

"""
Causal language model (+ LoRA) handler for CengBot
==================================================

Loads the active model (``models/active-model`` symlink, overridable with
``MODEL_PATH``) offline-first and generates short answers. Nothing here is
specific to one architecture: any decoder-only model that transformers loads
with ``AutoModelForCausalLM`` works (Llama, Qwen, DeepSeek distills, Mistral,
Gemma, Phi, GLM, Granite, OLMo, Falcon3, SmolLM, ...). The production model is
a LoRA adapter on ``meta-llama/Llama-3.2-3B``; the module keeps its historical
file name because it is imported everywhere.

Model directory contract (``models/<name>/``):
  1. PEFT LoRA adapter: ``adapter_config.json`` + ``adapter_model.safetensors``
     (or ``.bin``), either directly in the directory or in a legacy
     ``method1/`` sub-directory. The base model comes from
     ``adapter_config.json``'s ``base_model_name_or_path`` (fallback:
     ``BASE_MODEL_NAME``). The tokenizer is loaded from the model directory if
     it contains tokenizer files, otherwise from the base model.
  2. Full / merged Hugging Face model: ``config.json`` + ``*.safetensors``
     (or ``pytorch_model*.bin``) + tokenizer files.

Base model weights are looked up in ``MODEL_CACHE_DIR`` (default
``model_cache/``, Hugging Face cache layout) and in the default HF cache and are
loaded with ``local_files_only=True`` when present. A download is attempted
only if the weights are not cached and downloads are allowed (not
``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE`` and ``MODEL_ALLOW_DOWNLOAD`` not
false). ``LORA_MODEL_PATH`` is a legacy setting and is ignored.

Environment variables (all optional, read when the model is loaded):
  BASE_MODEL_NAME, MODEL_PATH, MODEL_CACHE_DIR, MODEL_TEMPERATURE,
  MODEL_MAX_NEW_TOKENS, MODEL_TOP_P, MODEL_TOP_K, MODEL_REPETITION_PENALTY,
  MODEL_PRECISION (bfloat16|float16|float32), USE_CUDA, MODEL_MAX_INPUT_TOKENS,
  MODEL_MAX_SENTENCES (0 = unlimited), MODEL_PROMPT_STYLE (training|legacy),
  MODEL_MERGE_LORA, MODEL_ALLOW_DOWNLOAD, HUGGING_FACE_TOKEN,
  MODEL_LOAD_IN_4BIT (default false; 4-bit NF4 base on CUDA for models that do
  not fit in bf16, e.g. >= ~12B on a 24 GB GPU; the LoRA then stays unmerged),
  MODEL_TRUST_REMOTE_CODE (default false; executes Python code shipped with the
  model repository - only for legacy remote-code repos such as chatglm3 or
  internlm2 that you trust).

Generation stops at every end-of-turn token the tokenizer/model knows (EOS,
``generation_config.eos_token_id`` and the family stop tokens in
``STOP_TOKEN_CANDIDATES``) and at generated turn markers. Reasoning blocks
(``<think>...</think>``) are never part of the answer.
"""

import gc
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel
from langdetect import DetectorFactory, LangDetectException, detect

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from error_handler import handle_model_error  # noqa: E402
import prompt_guard  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(_SRC_DIR).resolve().parent

# langdetect is non-deterministic unless seeded.
DetectorFactory.seed = 0

# Make sure .env values are visible even if this module is imported before
# config.env_loader (existing environment variables always win).
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:  # pragma: no cover - dotenv missing is not fatal
    pass


class ModelLoadError(RuntimeError):
    """Raised for invalid model directories / unavailable weights (clear, non-transient errors)."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Exactly the system prompts and turn markers the LoRA is trained with.
# src/train_model.build_prompt() imports these constants, so training and
# inference can never drift apart (guarded by a test).
TRAINING_SYSTEM_PROMPT_TR = (
    "Sen Çukurova Üniversitesi Bilgisayar Mühendisliği bölümünün deneyimli dijital asistanısın. "
    "Öğrencilere samimi, yardımsever ve doğru bilgiler vererek destek oluyorsun."
)
TRAINING_SYSTEM_PROMPT_EN = (
    "You are an experienced digital assistant for Cukurova University Computer Engineering Department. "
    "You provide friendly, helpful, and accurate information to support students."
)
TRAINING_TURN_MARKERS = {"tr": ("Öğrenci", "Asistan"), "en": ("Student", "Assistant")}

# Previous inference prompts (never seen during training); MODEL_PROMPT_STYLE=legacy.
LEGACY_SYSTEM_PROMPT_TR = """Sen Çukurova Üniversitesi Bilgisayar Mühendisliği bölümünün deneyimli dijital asistanısın. Öğrencilere samimi, yardımsever ve doğru bilgiler vererek destek oluyorsun.

Önemli kurallar:
- Her soruya MAKSİMUM 4 CÜMLE ile yanıt ver
- Cevapların çok kısa, net ve anlaşılır olmalı
- Gereksiz açıklamalardan ve tekrarlardan kesinlikle kaçın
- Selamlama mesajlarına tek cümlelik karşılık ver
- Sadece sorulan soruya odaklan, ekstra bilgi verme
- Cevaplarında MAKSİMUM 1 adet soru sorabilirsin
- Karşı tarafa çok fazla soru sorma, sadece gerektiğinde tek soru sor"""

LEGACY_SYSTEM_PROMPT_EN = """You are an experienced digital assistant for Çukurova University Computer Engineering Department. You help students by providing friendly, helpful and accurate information.

Important rules:
- Answer each question with MAXIMUM 4 SENTENCES
- Keep your answers very short, clear and understandable
- Absolutely avoid unnecessary explanations and repetitions
- Reply to greeting messages with a single sentence
- Focus only on the asked question, don't provide extra information
- You can ask MAXIMUM 1 question in your answers
- Don't ask too many questions, only ask one when necessary"""
LEGACY_TURN_MARKERS = {"tr": ("Student", "Assistant"), "en": ("Student", "Assistant")}

prompt_guard.register_protected_text(
    TRAINING_SYSTEM_PROMPT_TR, TRAINING_SYSTEM_PROMPT_EN, LEGACY_SYSTEM_PROMPT_TR, LEGACY_SYSTEM_PROMPT_EN
)

_FALLBACK_ANSWER = {
    "tr": "Üzgünüm, şu anda bir yanıt oluşturamadım. Lütfen sorunu farklı bir şekilde sormayı dene.",
    "en": "Sorry, I couldn't come up with an answer. Please try rephrasing your question.",
}
_ERROR_ANSWER = "Sorry, I encountered an error while generating a response. Please try again."


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, value, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    value = value.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    logger.warning("Invalid %s=%r, using default %s", name, value, default)
    return default


def resolve_project_path(path: str) -> Path:
    """Resolve ``path`` against the project root (never against the CWD)."""
    expanded = Path(os.path.expanduser(str(path)))
    return expanded if expanded.is_absolute() else PROJECT_ROOT / expanded


def downloads_allowed() -> bool:
    if _env_bool("HF_HUB_OFFLINE", False) or _env_bool("TRANSFORMERS_OFFLINE", False):
        return False
    return _env_bool("MODEL_ALLOW_DOWNLOAD", True)


class ModelConfig:
    """Model settings. Class attributes are the defaults; instances read the environment."""

    base_model = "meta-llama/Llama-3.2-3B"
    model_path = "models/active-model"
    lora_model_path = "models/active-model"  # legacy alias of model_path
    cache_dir = "model_cache"
    use_local_cache = True

    temperature = 0.7
    max_new_tokens = 200
    greeting_max_new_tokens = 30
    repetition_penalty = 1.1
    top_p = 0.95
    top_k = 50
    precision = "bfloat16"
    use_cuda = True
    max_input_tokens = 512  # LoRA was trained with max_length=512
    max_sentences = 4
    prompt_style = "training"
    merge_lora = True
    load_in_4bit = False
    trust_remote_code = False

    system_prompt_tr = TRAINING_SYSTEM_PROMPT_TR
    system_prompt_en = TRAINING_SYSTEM_PROMPT_EN

    hf_token = None

    def __init__(self):
        cls = type(self)
        self.base_model = _env_str("BASE_MODEL_NAME", cls.base_model)
        self.base_model_from_env = bool(os.getenv("BASE_MODEL_NAME", "").strip())
        self.model_path = str(resolve_project_path(_env_str("MODEL_PATH", cls.model_path)))
        self.lora_model_path = self.model_path
        self.cache_dir = str(resolve_project_path(_env_str("MODEL_CACHE_DIR", cls.cache_dir)))

        self.temperature = _env_float("MODEL_TEMPERATURE", cls.temperature)
        self.max_new_tokens = max(1, _env_int("MODEL_MAX_NEW_TOKENS", cls.max_new_tokens))
        self.greeting_max_new_tokens = min(cls.greeting_max_new_tokens, self.max_new_tokens)
        self.top_p = _env_float("MODEL_TOP_P", cls.top_p)
        self.top_k = _env_int("MODEL_TOP_K", cls.top_k)
        self.repetition_penalty = _env_float("MODEL_REPETITION_PENALTY", cls.repetition_penalty)
        self.precision = _env_str("MODEL_PRECISION", cls.precision).lower()
        self.use_cuda = _env_bool("USE_CUDA", cls.use_cuda)
        self.max_input_tokens = max(64, _env_int("MODEL_MAX_INPUT_TOKENS", cls.max_input_tokens))
        self.max_sentences = max(0, _env_int("MODEL_MAX_SENTENCES", cls.max_sentences))
        self.merge_lora = _env_bool("MODEL_MERGE_LORA", cls.merge_lora)
        self.load_in_4bit = _env_bool("MODEL_LOAD_IN_4BIT", cls.load_in_4bit)
        self.trust_remote_code = _env_bool("MODEL_TRUST_REMOTE_CODE", cls.trust_remote_code)

        style = _env_str("MODEL_PROMPT_STYLE", cls.prompt_style).lower()
        if style not in ("training", "legacy"):
            logger.warning("Invalid MODEL_PROMPT_STYLE=%r, using 'training'", style)
            style = "training"
        self.prompt_style = style
        if style == "legacy":
            self.system_prompt_tr, self.system_prompt_en = LEGACY_SYSTEM_PROMPT_TR, LEGACY_SYSTEM_PROMPT_EN
            self.turn_markers = dict(LEGACY_TURN_MARKERS)
        else:
            self.system_prompt_tr, self.system_prompt_en = TRAINING_SYSTEM_PROMPT_TR, TRAINING_SYSTEM_PROMPT_EN
            self.turn_markers = dict(TRAINING_TURN_MARKERS)

        self.hf_token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN") or None


_legacy_warning_emitted = False


def _warn_legacy_lora_path(active_model_path: str) -> None:
    """LORA_MODEL_PATH is ignored (models/active-model is the single source of truth)."""
    global _legacy_warning_emitted
    legacy = os.getenv("LORA_MODEL_PATH", "").strip()
    if not legacy or _legacy_warning_emitted:
        return
    legacy_real = os.path.realpath(resolve_project_path(legacy))
    active_real = os.path.realpath(active_model_path)
    if legacy_real in (active_real, os.path.join(active_real, "method1")):
        return
    _legacy_warning_emitted = True
    logger.warning(
        "LORA_MODEL_PATH=%s is a legacy setting and is IGNORED; the active model is %s "
        "(-> %s). Use scripts/switch_model.sh or MODEL_PATH instead and remove LORA_MODEL_PATH from .env.",
        legacy, active_model_path, active_real,
    )


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------

_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
_ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


@dataclass
class Source:
    """Something ``from_pretrained`` can load: a local directory or a hub id in an HF cache."""
    name: str                   # local directory or hub id passed to from_pretrained
    cache_dir: Optional[str]    # HF cache dir for hub ids (None for local directories)
    cached: bool                # files are present locally
    online_name: Optional[str]  # hub id for an online fallback (None for local directories)


@dataclass
class ModelSpec:
    kind: str                       # "lora" | "full"
    configured_path: str            # MODEL_PATH as configured (absolute)
    model_dir: str                  # real directory holding adapter / full weights
    version: str                    # e.g. "final-best-model-v1.2"
    base_model: Optional[str]       # base model id/path (lora only)
    base: Source                    # weights to load with AutoModelForCausalLM
    tokenizer_sources: List[Source]  # tried in order (offline), then online fallback

    @property
    def base_model_source(self) -> str:
        return self.base.name

    @property
    def base_local_files_only(self) -> bool:
        return self.base.cached

    @property
    def tokenizer_source(self) -> str:
        return self.tokenizer_sources[0].name


def _has_tokenizer(directory: Path) -> bool:
    return any((directory / name).is_file() for name in _TOKENIZER_FILES)


def _has_full_weights(directory: Path) -> bool:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        return bool(shards) and all((directory / shard).is_file() for shard in shards)
    return any(directory.glob("*.safetensors")) or any(directory.glob("pytorch_model*.bin"))


def _default_hf_cache() -> Optional[str]:
    try:
        from huggingface_hub import constants

        return constants.HF_HUB_CACHE
    except Exception:  # pragma: no cover
        return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def find_cached_snapshot(repo_id: str, cache_dirs: List[str], need_weights: bool = True,
                         need_tokenizer: bool = False) -> Optional[Tuple[Path, Path, bool]]:
    """Find a complete snapshot of ``repo_id`` in an HF cache.

    Returns (cache_dir, snapshot_dir, is_main_ref) or None.
    """
    for cache_dir in cache_dirs:
        if not cache_dir:
            continue
        repo_dir = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        main_snapshot = None
        ref = repo_dir / "refs" / "main"
        if ref.is_file():
            try:
                main_snapshot = snapshots / ref.read_text().strip()
            except OSError:
                main_snapshot = None
        candidates = [main_snapshot] if main_snapshot is not None else []
        candidates += sorted((p for p in snapshots.iterdir() if p.is_dir()),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in candidates:
            if not snap.is_dir():
                continue
            if need_weights and not ((snap / "config.json").is_file() and _has_full_weights(snap)):
                continue
            if need_tokenizer and not _has_tokenizer(snap):
                continue
            return Path(cache_dir), snap, snap == main_snapshot
    return None


def _looks_like_local_path(name: str) -> bool:
    """Hub ids look like "org/name"; anything else path-like is treated as a local path."""
    return name.startswith(("/", "./", "../", "~")) or name.count("/") > 1


def _same_model_name(a: str, b: str) -> bool:
    a_norm, b_norm = a.strip().rstrip("/").lower(), b.strip().rstrip("/").lower()
    return a_norm == b_norm or os.path.basename(a_norm) == os.path.basename(b_norm)


def _local_source(directory: Path) -> Source:
    return Source(name=str(directory), cache_dir=None, cached=True, online_name=None)


def _hub_source(name: str, config: ModelConfig, need_weights: bool, need_tokenizer: bool, what: str) -> Source:
    """Source for a hub id or local path; prefers MODEL_CACHE_DIR, then the default HF cache."""
    local = resolve_project_path(name)
    if _looks_like_local_path(name) or local.is_dir():
        if local.is_dir():
            return _local_source(local)
        raise ModelLoadError(f"{what} path '{name}' does not exist")
    found = find_cached_snapshot(name, [config.cache_dir, _default_hf_cache()],
                                 need_weights=need_weights, need_tokenizer=need_tokenizer)
    if found is None:
        # Not cached: the offline attempt (local_files_only=True, cache_dir=model_cache)
        # will fail fast and the loader falls back to a download if that is allowed.
        return Source(name=name, cache_dir=config.cache_dir, cached=False, online_name=name)
    cache_dir, snapshot, is_main = found
    if is_main:
        return Source(name=name, cache_dir=str(cache_dir), cached=True, online_name=name)
    # refs/main missing or pointing elsewhere: load the snapshot directory directly.
    return Source(name=str(snapshot), cache_dir=None, cached=True, online_name=name)


def _version_name(configured: Path) -> str:
    """Name of the real model directory (symlink resolved), e.g. "final-best-model-v1.2"."""
    return Path(os.path.realpath(configured)).name[:50]


def resolve_model_spec(config: Optional[ModelConfig] = None) -> ModelSpec:
    """Validate the configured model directory and work out how to load it (filesystem only)."""
    config = config or ModelConfig()
    configured = Path(config.model_path)
    if not configured.exists():
        if configured.is_symlink():
            raise ModelLoadError(
                f"Model path {configured} is a broken symlink (-> {os.readlink(configured)}). "
                "Point it at an existing model with scripts/switch_model.sh."
            )
        raise ModelLoadError(f"Model directory not found: {configured}")
    if not configured.is_dir():
        raise ModelLoadError(f"Model path is not a directory: {configured}")

    real_dir = Path(os.path.realpath(configured))

    # (1) LoRA adapter: the first COMPLETE layout wins - direct first, then legacy
    # method1/ (same rule as scripts/switch_model.sh validate, so a model that the
    # validator accepts always loads).
    adapter_dir = None
    incomplete = None
    for candidate in (real_dir, real_dir / "method1"):
        if (candidate / "adapter_config.json").is_file():
            if any((candidate / w).is_file() for w in _ADAPTER_WEIGHTS):
                adapter_dir = candidate
                break
            incomplete = incomplete or candidate
    if adapter_dir is None and incomplete is not None:
        raise ModelLoadError(
            f"{incomplete / 'adapter_config.json'} found but adapter weights are missing "
            f"(expected one of {', '.join(_ADAPTER_WEIGHTS)})"
        )

    if adapter_dir is not None:
        try:
            adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"Invalid {adapter_dir / 'adapter_config.json'}: {exc}") from exc
        if not isinstance(adapter_cfg, dict):
            raise ModelLoadError(f"Invalid {adapter_dir / 'adapter_config.json'}: not a JSON object")
        peft_type = str(adapter_cfg.get("peft_type", "LORA")).upper()
        if peft_type not in ("LORA", "ADALORA", "DORA"):
            logger.warning("Adapter %s has peft_type=%s (expected LORA)", adapter_dir, peft_type)

        adapter_base = (adapter_cfg.get("base_model_name_or_path") or "").strip()
        base_model = config.base_model
        if adapter_base:
            if _looks_like_local_path(adapter_base) and not resolve_project_path(adapter_base).is_dir():
                logger.warning("Adapter base model path '%s' does not exist here; using BASE_MODEL_NAME=%s",
                               adapter_base, config.base_model)
            else:
                base_model = adapter_base
                if config.base_model_from_env and not _same_model_name(adapter_base, config.base_model):
                    logger.warning(
                        "BASE_MODEL_NAME=%s differs from the adapter's base model %s; using the adapter's "
                        "base model (a LoRA only works with the base it was trained on).",
                        config.base_model, adapter_base,
                    )

        base = _hub_source(base_model, config, need_weights=True, need_tokenizer=False, what="Base model")
        tokenizers: List[Source] = []
        for directory in (adapter_dir, real_dir):
            if _has_tokenizer(directory) and all(t.name != str(directory) for t in tokenizers):
                tokenizers.append(_local_source(directory))
        tokenizers.append(_hub_source(base_model, config, need_weights=False, need_tokenizer=True,
                                      what="Base tokenizer"))
        return ModelSpec(
            kind="lora", configured_path=str(configured), model_dir=str(adapter_dir),
            version=_version_name(configured), base_model=base_model, base=base,
            tokenizer_sources=tokenizers,
        )

    # (2) Full / merged model
    if (real_dir / "config.json").is_file():
        if not _has_full_weights(real_dir):
            raise ModelLoadError(
                f"{real_dir / 'config.json'} found but model weights are missing or incomplete "
                "(expected *.safetensors or pytorch_model*.bin)"
            )
        tokenizers = []
        if _has_tokenizer(real_dir):
            tokenizers.append(_local_source(real_dir))
        else:
            logger.warning("Full model %s has no tokenizer files; using the tokenizer of %s",
                           real_dir, config.base_model)
        tokenizers.append(_hub_source(config.base_model, config, need_weights=False, need_tokenizer=True,
                                      what="Base tokenizer"))
        return ModelSpec(
            kind="full", configured_path=str(configured), model_dir=str(real_dir),
            version=_version_name(configured), base_model=None, base=_local_source(real_dir),
            tokenizer_sources=tokenizers,
        )

    raise ModelLoadError(
        f"{real_dir} is not a valid model directory: expected a LoRA adapter (adapter_config.json + "
        "adapter_model.safetensors, optionally in method1/) or a full model (config.json + *.safetensors)."
    )


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

class TurnMarkerStoppingCriteria(StoppingCriteria):
    """Stop as soon as the model starts a new dialogue turn (Student:/Öğrenci:/Assistant: ...)."""

    def __init__(self, tokenizer, prompt_length: int, window: int = 16):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.window = window
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs):
        done = []
        for row in input_ids:
            generated = row[self.prompt_length:]
            hit = False
            if generated.shape[0] > 0:
                tail = self.tokenizer.decode(generated[-self.window:], skip_special_tokens=True)
                hit = prompt_guard.find_turn_marker(tail) is not None
            done.append(hit)
        if any(done):
            self.triggered = True
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


_ABBREVIATIONS = {
    "dr", "prof", "doc", "yrd", "ars", "gor", "ogr", "uye", "vb", "vs", "orn", "bkz", "no", "st", "mr",
    "mrs", "ms", "eg", "ie", "etc", "yy", "sn", "av", "muh", "inc", "ltd", "co", "jr", "sr", "dk", "tel",
    "e.g", "i.e", "fak", "mah", "cad", "sok", "apt", "bl", "ref", "fig", "vol", "assoc", "asst", "dept",
}
_BOUNDARY_RE = re.compile(r"[.!?…]+(?=\s|$)")
_WORD_BEFORE_RE = re.compile(r"([^\s(\[\"']+)$")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_MARKER_PREFIXES = ("student", "öğrenci", "ogrenci", "assistant", "asistan", "user", "kullanıcı", "system",
                    "sistem", "human")


def _split_sentences(text: str) -> Tuple[List[str], int]:
    """Sentences (keeping trailing whitespace) and how many of the leading ones are *closed*:
    ended by punctuation that is followed by whitespace, so text generated later can no
    longer change them ("...cu." can still become "...cu.edu.tr", "...cu. " cannot)."""
    sentences: List[str] = []
    closed = 0
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        end = match.end()
        if match.group(0) == ".":
            before = _WORD_BEFORE_RE.search(text[start:match.start()])
            word = before.group(1) if before else ""
            folded = prompt_guard.fold_text(word).strip(".")
            if folded in _ABBREVIATIONS or (len(folded) == 1 and folded.isalpha()) or folded.isdigit():
                continue
        followed_by_space = end < len(text)  # _BOUNDARY_RE: whitespace or the end of the text
        while end < len(text) and text[end].isspace():
            end += 1
        if text[start:end].strip():
            sentences.append(text[start:end])
            if followed_by_space:
                closed = len(sentences)
        start = end
    if text[start:].strip():
        sentences.append(text[start:])
    return sentences, closed


def split_sentences(text: str) -> List[str]:
    """Split into sentences (keeping trailing whitespace), not breaking on "Prof. Dr." or "2. sınıf"."""
    return _split_sentences(text)[0]


def _select(sentences: List[str], limit: int) -> Tuple[List[int], Optional[int]]:
    """Indices of the kept sentences and of the rescued link sentence (see select_sentences)."""
    kept: List[int] = []
    dropped: List[int] = []
    asked_question = False
    for index, sentence in enumerate(sentences):
        is_question = sentence.rstrip().endswith("?")
        if (is_question and asked_question) or (limit and len(kept) >= limit):
            dropped.append(index)
            continue
        asked_question = asked_question or is_question
        kept.append(index)
    link = None
    if dropped and not any(_URL_RE.search(sentences[i]) for i in kept):
        link = next((i for i in dropped if _URL_RE.search(sentences[i])
                     and not sentences[i].rstrip().endswith("?")), None)
    return kept, link


def select_sentences(sentences: List[str], limit: int) -> Tuple[List[str], Optional[str]]:
    """What the post-processor keeps: the first ``limit`` sentences (0 = all) with at most one
    question, plus - if none of them has a link - the first link sentence among the dropped
    ones (links are the most useful part of many answers). Returns (kept, rescued_link)."""
    kept, link = _select(sentences, limit)
    return [sentences[i] for i in kept], (sentences[link] if link is not None else None)


def _ends_complete(sentence: str) -> bool:
    return sentence.rstrip().endswith((".", "!", "?", "…", ")", '"', "'", "»"))


def answer_is_final(text: str, max_sentences: int) -> bool:
    """True once more generated text can no longer change the post-processed answer.

    ``text`` is the decoded generation so far (reasoning and turn markers removed, trailing
    whitespace kept). The answer is final when ``max_sentences`` kept sentences are closed
    and either one of them has a link or a link sentence to append was already generated.
    Without a link the model gets a few more sentences to produce one: 4 after the last kept
    sentence (2 once any link appeared). Never before all kept sentences are complete.
    """
    if not max_sentences:
        return False
    sentences, closed = _split_sentences(text)
    closed_sentences = sentences[:closed]
    kept, link = _select(closed_sentences, max_sentences)
    if len(kept) < max_sentences:
        return False
    if link is not None or any(_URL_RE.search(closed_sentences[i]) for i in kept):
        return True
    patience = 2 if any(_URL_RE.search(s) for s in closed_sentences) else 4
    return closed - (kept[-1] + 1) >= patience


class SentenceLimitStoppingCriteria(StoppingCriteria):
    """Stop as soon as the post-processed answer is final (see ``answer_is_final``).

    The base model rarely emits EOS for Turkish and would otherwise run on to
    max_new_tokens, generating sentences that ``_postprocess`` drops. A sentence can only
    become closed when a token brings whitespace, so the decode + check runs only then.
    """

    def __init__(self, tokenizer, prompt_length: int, max_sentences: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.max_sentences = max_sentences
        self.triggered = False

    def _newest_brings_space(self, generated) -> bool:
        """Did the newest token add whitespace? Measured as a text delta: SentencePiece
        tokenizers (Llama 2, Mistral, Phi-3) strip the leading space when a single "▁word"
        token is decoded on its own, so decode([token]) would never show it."""
        last_two = self.tokenizer.decode(generated[-2:], skip_special_tokens=True)
        previous = self.tokenizer.decode(generated[-2:-1], skip_special_tokens=True) if generated.shape[0] > 1 else ""
        delta = last_two[len(previous):] if last_two.startswith(previous) else last_two
        return any(ch.isspace() for ch in delta)

    def _text(self, generated) -> str:
        # the same cleaning as _postprocess, but the trailing whitespace is kept
        text = prompt_guard.strip_reasoning(self.tokenizer.decode(generated, skip_special_tokens=True))
        return prompt_guard.cut_at_turn_markers(text, strip=False)

    def __call__(self, input_ids, scores, **kwargs):
        done = []
        for row in input_ids:
            generated = row[self.prompt_length:]
            hit = False
            if self.max_sentences and generated.shape[0] and self._newest_brings_space(generated):
                hit = answer_is_final(self._text(generated), self.max_sentences)
            done.append(hit)
        if any(done):
            self.triggered = True
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


def _strip_partial_marker(text: str) -> str:
    """Remove a dangling, partially generated turn marker at the end ("...\\nÖğr")."""
    lines = text.rstrip().split("\n")
    if len(lines) > 1:
        last = lines[-1].strip().lower()
        if last and any(prefix.startswith(last) for prefix in _MARKER_PREFIXES):
            return "\n".join(lines[:-1])
    return text


# End-of-turn / end-of-text tokens of the common open-weight families. A
# candidate is only used if the tokenizer really has it (added/special token
# with a real id), so e.g. Llama 3 gets <|end_of_text|> + <|eot_id|> only.
STOP_TOKEN_CANDIDATES = (
    "<|eot_id|>", "<|end_of_text|>",        # Llama 3, Granite, Falcon3
    "<|im_end|>", "<|endoftext|>",          # ChatML: Qwen, SmolLM, Phi-4; GPT-style EOS (Phi-3, GLM, OLMo)
    "</s>",                                 # SentencePiece EOS: Mistral, Llama 2
    "<end_of_turn>", "<eos>",               # Gemma
    "<|end|>", "<|user|>",                  # Phi-3 / GLM (a new user turn means the answer is over)
    "<|observation|>",                      # GLM
    "<｜end▁of▁sentence｜>",                 # DeepSeek (full-width bars)
)


def _int_ids(value) -> List[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value] if value >= 0 else []
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in _int_ids(item)]
    return []


def collect_stop_token_ids(tokenizer, model=None) -> List[int]:
    """All token ids that end an answer: the tokenizer's EOS, the model's
    ``generation_config.eos_token_id`` (int or list) and every entry of
    ``STOP_TOKEN_CANDIDATES`` that is a real token of this tokenizer."""
    ids: List[int] = []

    def add(value) -> None:
        for token_id in _int_ids(value):
            if token_id not in ids:
                ids.append(token_id)

    add(getattr(tokenizer, "eos_token_id", None))
    if model is not None:
        add(getattr(getattr(model, "generation_config", None), "eos_token_id", None))

    unk = getattr(tokenizer, "unk_token_id", None)
    try:
        added = tokenizer.get_added_vocab()
    except Exception:
        added = None
    specials = getattr(tokenizer, "all_special_tokens", None)
    known = set()
    if isinstance(added, dict):
        known.update(added)
    if isinstance(specials, (list, tuple, set)):
        known.update(specials)
    for token in STOP_TOKEN_CANDIDATES:
        # Real tokenizers: only added/special tokens (never a plain vocabulary
        # piece that happens to spell "</s>"). Minimal tokenizers without these
        # APIs: any id that is not unk.
        if (isinstance(added, dict) or isinstance(specials, (list, tuple, set))) and token not in known:
            continue
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0 and token_id != unk:
            add(token_id)
    return ids


def ensure_pad_token(tokenizer) -> None:
    """Tokenizers without a pad token (Llama, Mistral, Qwen base ...) pad with EOS."""
    if getattr(tokenizer, "pad_token", None) is not None:
        return
    if getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif getattr(tokenizer, "unk_token", None) is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        logger.warning("Tokenizer has neither a pad nor an EOS token; batching may fail")


def count_appended_special_tokens(tokenizer) -> int:
    """How many special tokens ``tokenizer(text)`` appends AFTER the text (e.g. an
    EOS added by a post-processor). BOS-style prefixes (Llama, Gemma, Mistral,
    GLM's [gMASK]<sop>) are fine; a suffix must be removed from prompts, or the
    model would see "end of text" right before it should answer. Training
    (train_model._appended_special_token_count) applies the same rule."""
    probe = "Merhaba, staj zorunlu mu?"
    try:
        with_special = list(tokenizer(probe, add_special_tokens=True, split_special_tokens=True)["input_ids"])
        plain = list(tokenizer(probe, add_special_tokens=False, split_special_tokens=True)["input_ids"])
    except Exception:
        return 0
    if not plain or len(with_special) <= len(plain):
        return 0
    for start in range(len(with_special) - len(plain) + 1):
        if with_special[start:start + len(plain)] == plain:
            return len(with_special) - start - len(plain)
    return 0


_GREETING_WORDS = {"selam", "selamlar", "merhaba", "meraba", "mrb", "mrhb", "slm", "sa", "hey", "gunaydin",
                   "hello", "hi", "hiya", "greetings", "hola", "naber", "nbr", "selamunaleykum"}
_GREETING_PHRASES = ("iyi gunler", "iyi aksamlar", "iyi geceler", "good morning", "good afternoon",
                     "good evening", "selamun aleykum", "selamin aleykum")


def is_greeting(message: str) -> bool:
    """True for short greeting-only messages ("Merhaba", "hi there!"), not for real questions."""
    folded = prompt_guard.fold_text(message)
    words = re.findall(r"[a-z0-9]+", folded)
    if not words or len(words) > 4:
        return False
    joined = " ".join(words)
    return any(w in _GREETING_WORDS for w in words) or any(p in joined for p in _GREETING_PHRASES)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CengBotModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = ModelConfig()
        self.device = torch.device("cuda" if (self.config.use_cuda and torch.cuda.is_available()) else "cpu")
        self.dtype = None
        self.spec: Optional[ModelSpec] = None
        self.load_time_s: Optional[float] = None
        self.last_load_error: Optional[str] = None
        self._load_failures = 0
        self._lock = threading.RLock()
        self._eos_ids: List[int] = []
        self._prompt_suffix_tokens = 0
        self._quantized_4bit = False

    # ---------------- language -----------------
    def detect_language(self, text: str) -> str:
        """Return 'tr' or 'en' (deterministic)."""
        text = (text or "").strip()
        if not text:
            return "en"
        guess_scores = self._language_scores(text)
        if guess_scores[0] != guess_scores[1]:
            return "tr" if guess_scores[0] > guess_scores[1] else "en"
        try:
            return "tr" if detect(text.lower()) == "tr" else "en"
        except LangDetectException:
            return "en"

    @staticmethod
    def _language_scores(text: str) -> Tuple[float, float]:
        tr_score = en_score = 0.0
        for word in prompt_guard._WORD_RE.findall(text):
            base = re.split(r"['’]", word)[0]
            folded = prompt_guard.fold_text(base)
            if folded == "cukurova":
                continue
            if any(c in prompt_guard._TR_CHARS for c in base):
                tr_score += 2.0 if (base[0].islower() or base.isupper()) else 0.5
            if folded in prompt_guard._TR_WORDS:
                tr_score += 1.0
            if folded in prompt_guard._EN_WORDS:
                en_score += 1.0
        return tr_score, en_score

    # ---------------- loading -----------------
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _resolve_device_and_dtype(self):
        cfg = self.config
        use_cuda = cfg.use_cuda and torch.cuda.is_available()
        if cfg.use_cuda and not use_cuda:
            logger.warning("USE_CUDA is enabled but CUDA is not available; running on CPU (slow).")
        precision = cfg.precision
        if not use_cuda:
            if precision not in ("float32", "fp32"):
                logger.info("CPU inference: using float32 instead of %s", precision)
            return torch.device("cpu"), torch.float32, None
        if precision in ("float32", "fp32"):
            dtype = torch.float32
        elif precision in ("float16", "fp16", "half"):
            dtype = torch.float16
        else:
            if precision not in ("bfloat16", "bf16"):
                logger.warning("Unknown MODEL_PRECISION=%r, using bfloat16", precision)
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                logger.warning("GPU does not support bfloat16; using float16")
                dtype = torch.float16
        return torch.device("cuda"), dtype, "auto"

    def load_model(self) -> bool:
        """Load tokenizer + model. Idempotent; returns False on failure (see ``last_load_error``)."""
        with self._lock:
            if self.is_loaded():
                return True
            self.config = ModelConfig()  # pick up env changes (e.g. after a model switch)
            _warn_legacy_lora_path(self.config.model_path)
            if self.config.trust_remote_code:
                logger.warning(
                    "SECURITY: MODEL_TRUST_REMOTE_CODE is enabled - Python code shipped with the model "
                    "repository will be EXECUTED while loading. Only use this for legacy remote-code "
                    "repositories you trust (e.g. chatglm3, internlm2); Llama, Qwen, Mistral, Gemma, Phi, "
                    "GLM-4 (-hf / 0414), Granite, OLMo ... are native in transformers and do not need it."
                )
            started = time.monotonic()
            try:
                spec = resolve_model_spec(self.config)
                device, dtype, device_map = self._resolve_device_and_dtype()
                logger.info("Loading %s model %s (base=%s, cached=%s, device=%s, dtype=%s)",
                            spec.kind, spec.model_dir, spec.base_model or "-", spec.base.cached, device,
                            str(dtype).replace("torch.", ""))

                tokenizer = self._load_tokenizer(spec)
                model = self._load_weights(spec, dtype, device_map)
                if device_map is None:
                    model.to(device)
                model.eval()

                self.model, self.tokenizer, self.spec = model, tokenizer, spec
                self.device, self.dtype = device, dtype
                self._eos_ids = self._collect_eos_ids(tokenizer, model)
                self._prompt_suffix_tokens = count_appended_special_tokens(tokenizer)
                logger.info("Stop token ids: %s%s", self._eos_ids,
                            f" (tokenizer appends {self._prompt_suffix_tokens} special token(s); "
                            "removed from prompts)" if self._prompt_suffix_tokens else "")
                self.load_time_s = round(time.monotonic() - started, 2)
                self.last_load_error = None
                self._load_failures = 0
                logger.info("Model %s loaded in %.1fs", spec.version, self.load_time_s)
                return True
            except Exception as exc:
                self._load_failures += 1
                self.last_load_error = f"{type(exc).__name__}: {exc}"
                self.model = self.tokenizer = None
                self._free_memory()
                if isinstance(exc, ModelLoadError):
                    logger.error("Failed to load model: %s", exc)
                elif self._load_failures == 1:
                    handle_model_error(exc, "load_model")  # full traceback, once
                    logger.error("Failed to load model: %s", self.last_load_error)
                else:
                    logger.error("Failed to load model (failure #%d): %s", self._load_failures,
                                 self.last_load_error[:500])
                return False

    def _offline_first(self, loader, sources: List[Source], what: str, **kwargs):
        """Call ``loader(name, local_files_only=True, cache_dir=...)`` for each source in order;
        only if every local attempt fails (and downloads are allowed) try online once."""
        local_errors: List[str] = []
        for source in sources:
            local_kwargs = dict(kwargs, local_files_only=True)
            if source.cache_dir:
                local_kwargs["cache_dir"] = source.cache_dir
            try:
                return loader(source.name, **local_kwargs)
            except OSError as exc:  # missing/incomplete local files (HF raises OSError subclasses)
                local_errors.append(f"{source.name}: {type(exc).__name__}: {str(exc)[:200]}")
            except ValueError as exc:
                self._raise_if_unsupported(exc, what, source.name)
                raise
        online_name = next((s.online_name for s in sources if s.online_name), None)
        detail = "; ".join(local_errors)
        if online_name is None:
            raise ModelLoadError(f"{what} could not be loaded from local files ({detail})")
        if not downloads_allowed():
            raise ModelLoadError(
                f"{what} '{online_name}' is not available offline ({detail}) and downloads are disabled "
                "(HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE / MODEL_ALLOW_DOWNLOAD). Copy the Hugging Face "
                f"cache into {self.config.cache_dir} or allow a one-time download."
            )
        logger.warning("%s '%s' is not available locally; downloading it from Hugging Face into %s",
                       what, online_name, self.config.cache_dir)
        online_kwargs = dict(kwargs, local_files_only=False, cache_dir=self.config.cache_dir)
        if self.config.hf_token:
            online_kwargs["token"] = self.config.hf_token
        try:
            return loader(online_name, **online_kwargs)
        except ValueError as exc:
            self._raise_if_unsupported(exc, what, online_name)
            raise

    @staticmethod
    def _raise_if_unsupported(exc: Exception, what: str, name: str) -> None:
        """Turn transformers' "custom code" / "unknown architecture" errors into clear ModelLoadErrors."""
        message = str(exc)
        if "trust_remote_code" in message:
            raise ModelLoadError(
                f"{what} '{name}' needs custom code from its repository. Prefer a checkpoint that is "
                "native in transformers (e.g. the '-hf' variant); otherwise set MODEL_TRUST_REMOTE_CODE=true "
                "(executes the repository's Python code - only for repositories you trust)."
            ) from exc
        if "does not recognize this architecture" in message:
            raise ModelLoadError(
                f"{what} '{name}': this transformers version does not support the architecture "
                f"({message[:200]}). Use a natively supported model or a newer transformers."
            ) from exc

    def _remote_code_kwargs(self) -> Dict[str, Any]:
        # Only passed when enabled, so the default call never mentions trust_remote_code.
        return {"trust_remote_code": True} if self.config.trust_remote_code else {}

    def _load_tokenizer(self, spec: ModelSpec):
        tokenizer = self._offline_first(AutoTokenizer.from_pretrained, spec.tokenizer_sources, "Tokenizer",
                                        use_fast=True, **self._remote_code_kwargs())
        ensure_pad_token(tokenizer)
        tokenizer.padding_side = "left"
        return tokenizer

    def _load_weights(self, spec: ModelSpec, dtype, device_map):
        kwargs: Dict[str, Any] = {"dtype": dtype, "low_cpu_mem_usage": True, **self._remote_code_kwargs()}
        if device_map is not None:
            kwargs["device_map"] = device_map
        self._quantized_4bit = False
        if self.config.load_in_4bit:
            if device_map is None:
                logger.warning("MODEL_LOAD_IN_4BIT needs a CUDA GPU (bitsandbytes); loading the model in %s",
                               str(dtype).replace("torch.", ""))
            else:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
                self._quantized_4bit = True
        model = self._offline_first(AutoModelForCausalLM.from_pretrained, [spec.base],
                                    "Base model" if spec.kind == "lora" else "Model", **kwargs)
        if spec.kind != "lora":
            return model
        base_model = model

        def load_adapter(name, **adapter_kwargs):
            adapter_kwargs.pop("cache_dir", None)
            return PeftModel.from_pretrained(base_model, name, is_trainable=False, **adapter_kwargs)

        try:
            model = self._offline_first(load_adapter, [_local_source(Path(spec.model_dir))], "LoRA adapter")
        except ModelLoadError:
            raise
        except (RuntimeError, ValueError, KeyError) as exc:
            raise ModelLoadError(
                f"LoRA adapter {spec.model_dir} does not match base model {spec.base_model}: {exc}"
            ) from exc
        if self.config.merge_lora and self._quantized_4bit:
            logger.info("4-bit base model: the LoRA adapter stays unmerged (merging into 4-bit weights loses precision)")
        elif self.config.merge_lora:
            # Fold the LoRA weights into the base weights: same maths, ~2x faster decoding.
            try:
                model = model.merge_and_unload()
            except Exception as exc:  # e.g. offloaded weights
                logger.warning("Could not merge LoRA weights (%s); using the unmerged adapter", exc)
        return model

    @staticmethod
    def _collect_eos_ids(tokenizer, model=None) -> List[int]:
        return collect_stop_token_ids(tokenizer, model)

    def _free_memory(self) -> None:
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    def unload_model(self) -> None:
        with self._lock:
            self.model = None
            self.tokenizer = None
            self.spec = None
            self._free_memory()

    def reload_model(self) -> bool:
        """Unload and load again (e.g. after switching models/active-model)."""
        with self._lock:
            self.unload_model()
            return self.load_model()

    def get_model_info(self) -> Dict[str, Any]:
        """Resolved model path/type/base model and load state (cheap; for health checks)."""
        info: Dict[str, Any] = {
            "loaded": self.is_loaded(),
            "configured_path": self.config.model_path,
            "prompt_style": self.config.prompt_style,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", "") if self.dtype is not None else None,
            "load_time_s": self.load_time_s,
            "last_error": self.last_load_error,
        }
        model = self.model
        if model is not None:
            info["architecture"] = getattr(getattr(model, "config", None), "model_type", None)
            info["stop_token_ids"] = list(self._eos_ids)
            info["load_in_4bit"] = self._quantized_4bit
        info["trust_remote_code"] = self.config.trust_remote_code
        spec = self.spec
        if spec is None:
            try:
                spec = resolve_model_spec(ModelConfig())
            except Exception as exc:
                info["error"] = str(exc)
        if spec is not None:
            info.update(
                resolved_path=spec.model_dir,
                model_type=spec.kind,
                version=spec.version,
                base_model=spec.base_model,
                base_model_source=spec.base_model_source,
                tokenizer_source=spec.tokenizer_source,
                base_cached=spec.base.cached,
            )
        return info

    # ---------------- prompting -----------------
    def format_prompt(self, message: str, language: Optional[str] = None) -> tuple:
        """Return (prompt, lang) using the same template the LoRA was trained with."""
        lang = language if language in ("tr", "en") else self.detect_language(message)
        logger.debug("Detected language: %s", lang)
        system_prompt = self.config.system_prompt_tr if lang == "tr" else self.config.system_prompt_en
        user_marker, assistant_marker = self.config.turn_markers[lang]
        return f"{system_prompt}\n\n{user_marker}: {message}\n{assistant_marker}:", lang

    def _encode(self, message: str, lang: str):
        """Tokenise the prompt; shorten the *message* (never the template) if it is too long."""
        message = prompt_guard.strip_special_tokens(message)
        limit = self.config.max_input_tokens
        suffix = getattr(self, "_prompt_suffix_tokens", 0)
        for _ in range(3):
            prompt, _ = self.format_prompt(message, lang)
            encoded = self.tokenizer(prompt, return_tensors="pt", split_special_tokens=True)
            if suffix:  # e.g. an EOS appended by the tokenizer: the prompt must end with "Assistant:"
                encoded = {k: (v[..., :-suffix] if k in ("input_ids", "attention_mask") else v)
                           for k, v in encoded.items()}
            length = encoded["input_ids"].shape[-1]
            if length <= limit:
                return encoded
            message_ids = self.tokenizer(message, add_special_tokens=False,
                                         split_special_tokens=True)["input_ids"]
            if hasattr(message_ids, "tolist"):
                message_ids = message_ids.tolist()
            keep = max(1, len(message_ids) - (length - limit) - 4)
            logger.warning("Question is too long (%d prompt tokens > %d); truncating it", length, limit)
            message = self.tokenizer.decode(message_ids[:keep], skip_special_tokens=True)
        return encoded

    def _input_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return getattr(self.model, "device", self.device)

    # ---------------- generation -----------------
    def generate_response(self, message: str) -> str:
        """Generate an answer (never raises; returns an apology on errors)."""
        return self.generate_response_with_info(message)[0]

    def generate_response_with_info(self, message: str, raise_errors: bool = False,
                                    language: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate an answer and return (answer, info). ``info`` has language, token counts, timing."""
        info: Dict[str, Any] = {"language": None, "input_tokens": None, "output_tokens": None,
                                "generation_time_s": None, "stopped_by": None,
                                "model_version": self.spec.version if self.spec else None}
        if not self.is_loaded():
            if raise_errors:
                raise RuntimeError("Model not loaded")
            return "Model not loaded.", info
        try:
            with self._lock:
                started = time.monotonic()
                lang = language if language in ("tr", "en") else self.detect_language(message)
                info["language"] = lang
                encoded = self._encode(message, lang)
                device = self._input_device()
                inputs = {k: v.to(device) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
                prompt_length = inputs["input_ids"].shape[-1]
                max_tokens = (self.config.greeting_max_new_tokens if is_greeting(message)
                              else self.config.max_new_tokens)
                sampling: Dict[str, Any] = {"do_sample": False}
                if self.config.temperature > 0:
                    sampling = {"do_sample": True, "temperature": self.config.temperature,
                                "top_p": self.config.top_p, "top_k": self.config.top_k}
                generation_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self._eos_ids or self.tokenizer.eos_token_id,
                    **sampling,
                )
                stopper = TurnMarkerStoppingCriteria(self.tokenizer, prompt_length)
                sentence_stopper = SentenceLimitStoppingCriteria(self.tokenizer, prompt_length,
                                                                 self.config.max_sentences)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=generation_config,
                        stopping_criteria=StoppingCriteriaList([stopper, sentence_stopper]),
                    )
                new_tokens = outputs[0][prompt_length:]
                raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                if stopper.triggered:
                    stopped_by = "turn_marker"
                elif sentence_stopper.triggered:
                    stopped_by = "sentence_limit"
                elif int(new_tokens.shape[-1]) >= max_tokens:
                    stopped_by = "max_tokens"
                else:
                    stopped_by = "eos"
                answer = self._postprocess(raw, lang, truncated=stopped_by in ("max_tokens", "sentence_limit"))
                info.update(
                    input_tokens=int(prompt_length),
                    output_tokens=int(new_tokens.shape[-1]),
                    generation_time_s=round(time.monotonic() - started, 3),
                    stopped_by=stopped_by,
                )
                return answer, info
        except Exception as exc:
            self._free_memory()
            if raise_errors:
                raise
            handle_model_error(exc, "generate_response")
            logger.error("Generation error: %s", exc)
            return _ERROR_ANSWER, info

    def _postprocess(self, raw: str, lang: str, truncated: bool = False) -> str:
        text = prompt_guard.strip_reasoning(raw)  # <think>...</think> of reasoning models
        text = prompt_guard.cut_at_end_of_turn(text)
        text = prompt_guard.strip_special_tokens(text)
        text = prompt_guard.cut_at_turn_markers(text)
        text = _strip_partial_marker(text).strip()
        if not text:
            return _FALLBACK_ANSWER.get(lang, _FALLBACK_ANSWER["en"])

        sentences = split_sentences(text)
        if truncated and len(sentences) > 1 and not _ends_complete(sentences[-1]):
            sentences = sentences[:-1]  # drop the sentence cut off by max_new_tokens

        kept, link = select_sentences(sentences, self.config.max_sentences)
        if link is not None:
            kept.append(link)
        text = "".join(kept).strip()
        if text and text[-1].isalnum():
            text += "."
        return text


# Global model instance
model_instance = CengBotModel()

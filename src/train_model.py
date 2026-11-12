#!/usr/bin/env python3
"""
CengBot LoRA fine-tuning pipeline
=================================

Data flow::

    university_bot.db --(database_to_training.py)--> data/cengbot_qa_augmented.jsonl
    data/merged_qa_data.jsonl (base corpus)  ------+
                                                   +--> this script --> models/<name>/

Guarantees (covered by tests/test_training_pipeline.py):

* The training prompt is byte-identical to the inference prompt built by
  ``llama_model_handler.CengBotModel.format_prompt()`` (default
  ``MODEL_PROMPT_STYLE=training``)::

      TR: "{system_prompt_tr}\\n\\nÖğrenci: {message}\\nAsistan:"
      EN: "{system_prompt_en}\\n\\nStudent: {message}\\nAssistant:"

  followed by ``" {answer}<eos>"``. System prompts and turn markers are taken
  from ``llama_model_handler.TRAINING_SYSTEM_PROMPT_*`` /
  ``TRAINING_TURN_MARKERS`` (single source of truth).
* Loss is computed on the answer tokens and the EOS token only (the prompt is
  masked with -100), so the model learns to stop. The prompt is tokenized
  exactly like at inference time (``tokenizer(prompt, split_special_tokens=True)``,
  with BOS).
* Examples longer than ``--max-length`` are dropped (or truncated with
  ``--truncation truncate``) and always reported - never cut silently.
* The train/validation split is group-aware: an original question and its
  augmented variants (and exact duplicates) never end up on both sides.
* The final model directory follows the model directory contract:
  ``adapter_config.json`` + ``adapter_model.safetensors`` + tokenizer files +
  ``training_info.json`` directly in ``models/<name>/`` (no ``method1/``, no
  optimizer state). Checkpoints live in ``checkpoints/<name>/`` instead.
* The base model is loaded offline-first from ``model_cache/``. Any
  decoder-only architecture that transformers supports natively works (Llama,
  Qwen, DeepSeek distills, Mistral, Gemma, Phi, GLM, Granite, OLMo, Falcon3,
  SmolLM, ...): ``--lora-target-modules auto`` (default) uses the classic
  q/k/v/o/gate/up/down projections when the model has all of them and PEFT's
  ``all-linear`` otherwise (fused ``qkv_proj``/``gate_up_proj`` of Phi-3/4 and
  GLM). The choice is recorded in ``training_info.json``.

Usage::

    python3 src/train_model.py --dry-run     # validate data + tokenization, no model
    python3 src/train_model.py               # full training run
    python3 src/train_model.py --base-model Qwen/Qwen2.5-7B --load-in-4bit --output-name qwen25-7b-v1
    python3 src/train_model.py --resume-from-checkpoint latest   # continue an interrupted run
    # smoke run, every output outside the project:
    python3 src/train_model.py --dataset small.jsonl --max-steps 30 --grad-accum 1 \\
        --models-dir /tmp/x/models --checkpoint-dir /tmp/x/ckpt --log-dir /tmp/x/logs --no-db-export
    python3 src/train_model.py --help

Author: naholav
"""

import argparse
import csv
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

logger = logging.getLogger("train_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B"
DEFAULT_DATASET_FILES = ("data/merged_qa_data.jsonl", "data/cengbot_qa_augmented.jsonl")
DB_EXPORT_FILENAME = "cengbot_qa_augmented.jsonl"
MODEL_DIR_PREFIX = "final-best-model-v"
ACTIVE_MODEL_LINK = "active-model"
IGNORE_INDEX = -100
MIN_EXAMPLES_FOR_VALIDATION = 10
# Inference generates at most this many new tokens by default
# (llama_model_handler.ModelConfig.max_new_tokens); longer answers are reported
# because they can never be reproduced in full.
DEFAULT_INFERENCE_MAX_NEW_TOKENS = 200

# Must stay identical to llama_model_handler.CengBotModel.format_prompt()
# (MODEL_PROMPT_STYLE=training, the inference default).
PROMPT_TEMPLATE = "{system_prompt}\n\n{user_marker}: {message}\n{assistant_marker}:"
# The answer follows "Assistant:"/"Asistan:" after a single space (the model generates it).
ANSWER_PREFIX = " "

# Fallback copies of llama_model_handler.TRAINING_SYSTEM_PROMPT_* /
# TRAINING_TURN_MARKERS. Only used when llama_model_handler cannot be imported
# (a warning is logged in that case).
_FALLBACK_SYSTEM_PROMPT_TR = (
    "Sen Çukurova Üniversitesi Bilgisayar Mühendisliği bölümünün deneyimli dijital asistanısın. "
    "Öğrencilere samimi, yardımsever ve doğru bilgiler vererek destek oluyorsun."
)
_FALLBACK_SYSTEM_PROMPT_EN = (
    "You are an experienced digital assistant for Cukurova University Computer Engineering Department. "
    "You provide friendly, helpful, and accurate information to support students."
)
_FALLBACK_TURN_MARKERS = {"tr": ("Öğrenci", "Asistan"), "en": ("Student", "Assistant")}
# Inference tokenizes with split_special_tokens=True (special-token strings in
# user text are plain text) - training must do the same.
TOKENIZER_CALL_KWARGS = {"split_special_tokens": True}

# Explicit LoRA targets (attention + MLP projections of Llama-style models:
# Llama, Qwen, Mistral, Gemma, Granite, OLMo ...). ``auto`` uses them when the
# model has all of them, PEFT's "all-linear" otherwise.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_TARGET_AUTO = "auto"
LORA_TARGET_ALL_LINEAR = "all-linear"
# transformers recommends eager attention for training these (attention logit soft-capping).
EAGER_ATTENTION_MODEL_TYPES = ("gemma2", "gemma3", "gemma3_text")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Paths (all derived from the project root - no hardcoded machine paths)
    base_path: str = str(PROJECT_ROOT)
    dataset_paths: List[str] = field(
        default_factory=lambda: [str(PROJECT_ROOT / p) for p in DEFAULT_DATASET_FILES]
    )
    output_dir: str = str(PROJECT_ROOT / "models")  # final model dirs go here
    checkpoint_root: str = str(PROJECT_ROOT / "checkpoints")  # Trainer checkpoints
    log_dir: str = str(PROJECT_ROOT / "logs")
    cache_dir: str = str(PROJECT_ROOT / "model_cache")
    output_name: Optional[str] = None  # default: final-best-model-v<next>

    # Model
    model_name: str = DEFAULT_BASE_MODEL

    # Training (effective batch size = 8 * 8 = 64). The micro-batch split does not
    # change the step's gradient (summed over the micro-batches, normalised by the
    # step's answer tokens); 8 x 8 instead of 4 x 16 cut the RTX 3090 step time
    # from 6.4 s to 4.4 s at 10.9 GiB peak (fits next to the ~7 GB inference worker).
    num_epochs: int = 4
    max_steps: int = -1  # > 0: stop after this many optimizer steps (overrides num_epochs)
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_length: int = 512
    truncation_policy: str = "drop"  # "drop" or "truncate"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Any = LORA_TARGET_AUTO  # "auto" | "all-linear" | list of module names

    # Validation
    val_split_ratio: float = 0.07
    early_stopping_patience: int = 3
    evals_per_epoch: int = 5

    # Misc
    seed: int = 42
    load_in_4bit: bool = False  # QLoRA (inference uses a bf16 base model)
    trust_remote_code: bool = False  # execute model-repo code (legacy remote-code models only)
    force_cpu: bool = False
    semantic_dedup: bool = False
    export_from_db: bool = True
    # The data is tokenized in memory and collation is trivial: worker processes
    # measured no faster (0: 5.97, 2: 6.22, 4: 6.09 s/step on the RTX 3090).
    dataloader_num_workers: int = 0
    sort_by_length: bool = True  # micro-batches of similar length inside each step (StepSortedRandomSampler)
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    overwrite_output: bool = False

    @property
    def dataset_path(self) -> str:
        """Backward compatible single dataset path (the first one)."""
        return self.dataset_paths[0] if self.dataset_paths else ""

    @property
    def history_dir(self) -> str:
        return str(Path(self.log_dir) / "training_history")


config = Config()


# ---------------------------------------------------------------------------
# Prompt format (shared with inference)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_prompt_spec() -> Dict[str, Any]:
    """System prompts + turn markers used at inference (single source of truth:
    ``llama_model_handler.TRAINING_SYSTEM_PROMPT_*`` / ``TRAINING_TURN_MARKERS``)."""
    spec = {
        "system": {"tr": _FALLBACK_SYSTEM_PROMPT_TR, "en": _FALLBACK_SYSTEM_PROMPT_EN},
        "markers": dict(_FALLBACK_TURN_MARKERS),
        "max_new_tokens": DEFAULT_INFERENCE_MAX_NEW_TOKENS,
        "source": "train_model fallback copy",
    }
    try:
        import llama_model_handler as lmh  # lazy: heavy import, only needed for the prompt spec

        system = {"tr": getattr(lmh, "TRAINING_SYSTEM_PROMPT_TR", None), "en": getattr(lmh, "TRAINING_SYSTEM_PROMPT_EN", None)}
        markers = getattr(lmh, "TRAINING_TURN_MARKERS", None)
        if (
            all(isinstance(v, str) and v.strip() for v in system.values())
            and isinstance(markers, dict)
            and all(isinstance(markers.get(k), (tuple, list)) and len(markers[k]) == 2 for k in ("tr", "en"))
        ):
            spec.update(system=system, markers={k: tuple(markers[k]) for k in ("tr", "en")},
                        source="llama_model_handler.TRAINING_*")
            max_new = getattr(getattr(lmh, "ModelConfig", None), "max_new_tokens", None)
            if isinstance(max_new, int) and max_new > 0:
                spec["max_new_tokens"] = max_new
        else:
            logger.warning("llama_model_handler has no usable TRAINING_* prompt constants; using built-in copies")
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning(
            "Could not import the inference prompt spec from llama_model_handler (%s); "
            "using built-in copies - verify they still match inference!",
            exc,
        )
    return spec


def get_system_prompts() -> Tuple[str, str, str]:
    """Return ``(system_prompt_tr, system_prompt_en, source)``."""
    spec = get_prompt_spec()
    return spec["system"]["tr"], spec["system"]["en"], spec["source"]


def _strip_special_tokens(text: str) -> str:
    """Same user-message sanitizing as inference (prompt_guard), identity if unavailable."""
    try:
        import prompt_guard
    except Exception:  # pragma: no cover
        return text
    return prompt_guard.strip_special_tokens(text)


_LANGUAGE_ALIASES = {
    "tr": "turkish",
    "tur": "turkish",
    "turkish": "turkish",
    "türkçe": "turkish",
    "turkce": "turkish",
    "en": "english",
    "eng": "english",
    "english": "english",
    "ingilizce": "english",
    "i̇ngilizce": "english",
}


def normalize_language(value: Any) -> Optional[str]:
    """Map 'TR'/'tr'/'turkish'/... -> 'turkish' and 'EN'/... -> 'english'."""
    if not isinstance(value, str):
        return None
    return _LANGUAGE_ALIASES.get(value.strip().lower())


def build_prompt(question: str, language: str) -> str:
    """Build the exact inference prompt for ``question`` (without the answer)."""
    spec = get_prompt_spec()
    lang = "tr" if normalize_language(language) == "turkish" else "en"
    user_marker, assistant_marker = spec["markers"][lang]
    return PROMPT_TEMPLATE.format(
        system_prompt=spec["system"][lang],
        user_marker=user_marker,
        assistant_marker=assistant_marker,
        message=_strip_special_tokens(question),
    )


def format_training_text(question: str, answer: str, language: str) -> str:
    """Prompt + answer as plain text (EOS is added at token level)."""
    return build_prompt(question, language) + ANSWER_PREFIX + answer


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_format_info() -> Dict[str, Any]:
    spec = get_prompt_spec()
    return {
        "template": PROMPT_TEMPLATE,
        "answer_prefix": ANSWER_PREFIX,
        "turn_markers": {k: list(v) for k, v in spec["markers"].items()},
        "inference_prompt_style": "training",
        "system_prompt_source": spec["source"],
        "system_prompt_sha256": {"turkish": _sha256_text(spec["system"]["tr"]), "english": _sha256_text(spec["system"]["en"])},
        "split_special_tokens": True,
        "eos_appended": True,
        "loss_on": "answer_and_eos_only",
    }


# ---------------------------------------------------------------------------
# Dataset loading / cleaning / dedup
# ---------------------------------------------------------------------------
_DOTLESS_FOLD = str.maketrans({"İ": "i", "I": "i", "ı": "i"})


def _norm_text(text: str) -> str:
    """Case/whitespace-insensitive key; Turkish dotted/dotless i folded together."""
    return " ".join(text.translate(_DOTLESS_FOLD).casefold().split())


def load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file (UTF-8, BOM tolerant). Invalid lines are skipped and reported."""
    records: List[Dict[str, Any]] = []
    invalid = 0
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid += 1
                if invalid <= 5:
                    logger.warning("%s:%d: invalid JSON skipped (%s)", path, lineno, exc)
                continue
            if not isinstance(obj, dict):
                invalid += 1
                continue
            obj["_file"] = str(path)
            obj["_line"] = lineno
            records.append(obj)
    if invalid:
        logger.warning("%s: skipped %d invalid line(s)", path, invalid)
    return records


def clean_records(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Validate, normalize and exactly-deduplicate records.

    Output records have ``question``/``answer`` (stripped), ``language``
    ('turkish'/'english'), optional ``group_id`` and the private ``_file``/
    ``_line`` provenance used for grouping.
    """
    cleaned: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    seen = set()
    for rec in records:
        question, answer = rec.get("question"), rec.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str) or not question.strip() or not answer.strip():
            stats["missing_text"] += 1
            continue
        language = normalize_language(rec.get("language"))
        if language is None:
            stats["unknown_language"] += 1
            continue
        question, answer = question.strip(), answer.strip()
        key = (language, _norm_text(question), _norm_text(answer))
        if key in seen:
            stats["exact_duplicate"] += 1
            continue
        seen.add(key)
        item: Dict[str, Any] = {"question": question, "answer": answer, "language": language}
        group = rec.get("group_id")
        if group not in (None, ""):
            item["group_id"] = str(group)
        item["_file"] = rec.get("_file")
        item["_line"] = rec.get("_line")
        cleaned.append(item)
    stats["kept"] = len(cleaned)
    if stats.get("missing_text") or stats.get("unknown_language"):
        logger.warning(
            "Dropped %d record(s) without question/answer and %d with unknown language",
            stats.get("missing_text", 0),
            stats.get("unknown_language", 0),
        )
    return cleaned, dict(stats)


def load_training_records(dataset_paths: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load + clean all existing dataset files. Raises if none exists."""
    raw: List[Dict[str, Any]] = []
    files_info = []
    for path in dataset_paths:
        if not os.path.isfile(path):
            logger.warning("Dataset file not found, skipping: %s", path)
            continue
        recs = load_jsonl_records(path)
        files_info.append({"path": str(path), "records": len(recs), "sha256": _sha256_file(path)})
        logger.info("Loaded %d records from %s", len(recs), path)
        raw.extend(recs)
    if not files_info:
        raise FileNotFoundError(f"None of the dataset files exist: {list(dataset_paths)}")
    cleaned, stats = clean_records(raw)
    return cleaned, {"files": files_info, "raw_records": len(raw), **stats}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def semantic_deduplicate(
    records: List[Dict[str, Any]],
    cache_dir: Optional[str] = None,
    answer_threshold: float = 0.94,
    question_threshold: float = 0.98,
    encoder=None,
) -> List[Dict[str, Any]]:
    """Optional near-duplicate removal (same language, very similar Q *and* A).

    Uses a multilingual sentence-transformer, loaded offline-first. Keeps the
    first occurrence. Vectorized: O(n^2) dot products but no Python n^2 loop.
    """
    if not records:
        return records
    if encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("sentence-transformers not installed - skipping semantic dedup")
            return records
        name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        try:
            encoder = SentenceTransformer(name, cache_folder=cache_dir, local_files_only=True)
        except Exception:
            if not downloads_allowed():
                logger.warning("Semantic dedup model not cached and downloads are disabled - skipping")
                return records
            try:
                encoder = SentenceTransformer(name, cache_folder=cache_dir)
            except Exception as exc:
                logger.warning("Could not load semantic dedup model (%s) - skipping", exc)
                return records

    import numpy as np

    keep = [True] * len(records)
    by_lang: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_lang[rec["language"]].append(idx)
    for lang, idxs in by_lang.items():
        if len(idxs) < 2:
            continue
        q = np.asarray(encoder.encode([records[i]["question"] for i in idxs], batch_size=64,
                                      show_progress_bar=False, normalize_embeddings=True))
        a = np.asarray(encoder.encode([records[i]["answer"] for i in idxs], batch_size=64,
                                      show_progress_bar=False, normalize_embeddings=True))
        removed = np.zeros(len(idxs), dtype=bool)
        for i in range(len(idxs) - 1):
            if removed[i]:
                continue
            sim_a = a[i + 1:] @ a[i]
            sim_q = q[i + 1:] @ q[i]
            dup = np.nonzero((sim_a >= answer_threshold) & (sim_q >= question_threshold))[0] + i + 1
            removed[dup] = True
        for pos, is_removed in enumerate(removed):
            if is_removed:
                keep[idxs[pos]] = False
        logger.info("Semantic dedup %s: removed %d of %d", lang, int(removed.sum()), len(idxs))
    return [rec for rec, k in zip(records, keep) if k]


# ---------------------------------------------------------------------------
# Leakage-free train/validation split
# ---------------------------------------------------------------------------
_COURSE_CODE_RE = re.compile(r"\b([A-ZÇĞİÖŞÜ]{2,4})\s?(\d{3})\b")


def assign_groups(records: Sequence[Dict[str, Any]], link_course_codes: bool = True) -> List[int]:
    """Assign a group id to every record; records in one group are "the same question".

    Linked together (union-find):
    * records with the same explicit ``group_id`` (written by data_augmentation /
      database_to_training: an original and its augmented variants);
    * legacy augmented files without ``group_id``: consecutive records of the
      same language in the same file (original followed by its variations);
    * records with the same normalized question or the same normalized answer;
    * records mentioning the same course code (e.g. CEN216) - catches the EN/TR
      variants of the same fact.
    """
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first_by_key: Dict[Tuple[str, str], int] = {}

    def link(key: Tuple[str, str], idx: int) -> None:
        first = first_by_key.setdefault(key, idx)
        if first != idx:
            union(first, idx)

    for i, rec in enumerate(records):
        gid = rec.get("group_id")
        if gid is not None:
            link(("group", str(gid)), i)
        elif i > 0:
            prev = records[i - 1]
            if (
                prev.get("group_id") is None
                and rec.get("_file") is not None
                and prev.get("_file") == rec.get("_file")
                and prev.get("language") == rec.get("language")
            ):
                union(i - 1, i)
        link(("q", _norm_text(rec["question"])), i)
        link(("a", _norm_text(rec["answer"])), i)
        if link_course_codes:
            text = rec["question"] + " " + rec["answer"]
            for m in _COURSE_CODE_RE.finditer(text):
                link(("code", m.group(1) + m.group(2)), i)

    remap: Dict[int, int] = {}
    return [remap.setdefault(find(i), len(remap)) for i in range(n)]


def split_train_validation(
    records: Sequence[Dict[str, Any]],
    val_ratio: float,
    seed: int = 42,
    groups: Optional[Sequence[int]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Group-aware random split. No group is ever on both sides."""
    n = len(records)
    if n == 0:
        return [], []
    if groups is None:
        groups = assign_groups(records)
    members: Dict[int, List[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        members[g].append(idx)

    target = int(round(n * val_ratio)) if val_ratio > 0 and n >= MIN_EXAMPLES_FOR_VALIDATION else 0
    rng = random.Random(seed)
    group_ids = sorted(members)
    rng.shuffle(group_ids)

    # Stratify by language: each group counts for its majority language and every
    # language gets ~val_ratio of its examples (like the old per-language split).
    def majority_language(g: int) -> str:
        langs = Counter(records[i].get("language") for i in members[g])
        return sorted(langs.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]

    group_lang = {g: majority_language(g) for g in group_ids}
    lang_total: Counter = Counter()
    for g in group_ids:
        lang_total[group_lang[g]] += len(members[g])
    lang_target = {lang: int(round(count * val_ratio)) for lang, count in lang_total.items()}

    val_idx: set = set()
    lang_size: Counter = Counter()
    if target > 0:
        for g in group_ids:
            lang = group_lang[g]
            m = members[g]
            if lang_size[lang] + len(m) <= lang_target[lang]:
                val_idx.update(m)
                lang_size[lang] += len(m)
        if not val_idx:
            smallest = min(group_ids, key=lambda g: (len(members[g]), g))
            if len(members[smallest]) * 2 <= n:
                val_idx.update(members[smallest])
        if not val_idx:
            logger.warning(
                "Could not build a leakage-free validation split (all %d examples are in "
                "large groups) - training without validation",
                n,
            )
    elif val_ratio > 0:
        logger.warning("Only %d examples - training without a validation set", n)

    train = [records[i] for i in range(n) if i not in val_idx]
    val = [records[i] for i in sorted(val_idx)]
    rng.shuffle(train)
    return train, val


def split_by_language(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tr = [r for r in records if r.get("language") == "turkish"]
    en = [r for r in records if r.get("language") == "english"]
    return tr, en


# ---------------------------------------------------------------------------
# Tokenization with answer-only loss
# ---------------------------------------------------------------------------
def _assemble_example(
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
    eos_token_id: int,
    max_length: int,
    truncation_policy: str = "drop",
) -> Tuple[Optional[Dict[str, List[int]]], str]:
    prompt_ids = list(prompt_ids)
    answer_ids = list(answer_ids) + [eos_token_id]
    if len(prompt_ids) >= max_length:
        return None, "prompt_too_long"
    status = "ok"
    if len(prompt_ids) + len(answer_ids) > max_length:
        if truncation_policy == "truncate":
            answer_ids = answer_ids[: max_length - len(prompt_ids)]
            status = "truncated"
        else:
            return None, "too_long"
    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}, status


def _eos_id(tokenizer) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("Tokenizer has no eos_token_id - cannot teach the model to stop "
                         "(set eos_token in the tokenizer_config.json of the base model)")
    return int(eos)


def _appended_special_token_count(tokenizer) -> int:
    """Number of special tokens ``tokenizer(text)`` appends after the text (e.g. an EOS
    added by a post-processor). Prompts are cut before them - same rule as inference
    (llama_model_handler.count_appended_special_tokens). BOS-style prefixes stay."""
    probe = "Merhaba, staj zorunlu mu?"
    try:
        with_special = list(tokenizer(probe, add_special_tokens=True, **TOKENIZER_CALL_KWARGS)["input_ids"])
        plain = list(tokenizer(probe, add_special_tokens=False, **TOKENIZER_CALL_KWARGS)["input_ids"])
    except Exception:
        return 0
    if not plain or len(with_special) <= len(plain):
        return 0
    for start in range(len(with_special) - len(plain) + 1):
        if with_special[start:start + len(plain)] == plain:
            return len(with_special) - start - len(plain)
    return 0


def _strip_suffix(ids: Sequence[int], suffix: int) -> List[int]:
    ids = list(ids)
    return ids[:-suffix] if suffix else ids


def tokenizer_info(tokenizer) -> Dict[str, Any]:
    """Special-token facts of the tokenizer (recorded in training_info.json)."""
    probe_ids = list(tokenizer("x", add_special_tokens=True, **TOKENIZER_CALL_KWARGS)["input_ids"])
    bos = getattr(tokenizer, "bos_token_id", None)
    return {
        "class": type(tokenizer).__name__,
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "adds_bos": bool(bos is not None and probe_ids and probe_ids[0] == bos),
        "appended_special_tokens_removed": _appended_special_token_count(tokenizer),
    }


def encode_example(
    tokenizer,
    question: str,
    answer: str,
    language: str,
    max_length: int,
    truncation_policy: str = "drop",
) -> Tuple[Optional[Dict[str, List[int]]], str]:
    """Tokenize one example exactly like inference + answer + EOS.

    Returns ``(features or None, status)`` where status is one of
    ``ok``/``truncated``/``too_long``/``prompt_too_long``.
    """
    prompt_ids = tokenizer(build_prompt(question, language), add_special_tokens=True, **TOKENIZER_CALL_KWARGS)["input_ids"]
    prompt_ids = _strip_suffix(prompt_ids, _appended_special_token_count(tokenizer))
    answer_ids = tokenizer(ANSWER_PREFIX + answer, add_special_tokens=False, **TOKENIZER_CALL_KWARGS)["input_ids"]
    return _assemble_example(prompt_ids, answer_ids, _eos_id(tokenizer), max_length, truncation_policy)


def tokenize_records(
    records: Sequence[Dict[str, Any]],
    tokenizer,
    max_length: int,
    truncation_policy: str = "drop",
    batch_size: int = 1000,
    name: str = "dataset",
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    """Batch-tokenize records. Every dropped/truncated example is counted and reported."""
    eos = _eos_id(tokenizer)
    suffix = _appended_special_token_count(tokenizer)
    encoded: List[Dict[str, List[int]]] = []
    stats: Counter = Counter()
    dropped_examples: List[str] = []
    answer_lengths: List[int] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        prompts = [build_prompt(r["question"], r["language"]) for r in chunk]
        answers = [ANSWER_PREFIX + r["answer"] for r in chunk]
        prompt_ids = tokenizer(prompts, add_special_tokens=True, **TOKENIZER_CALL_KWARGS)["input_ids"]
        answer_ids = tokenizer(answers, add_special_tokens=False, **TOKENIZER_CALL_KWARGS)["input_ids"]
        for rec, p_ids, a_ids in zip(chunk, prompt_ids, answer_ids):
            features, status = _assemble_example(_strip_suffix(p_ids, suffix), a_ids, eos, max_length, truncation_policy)
            stats[status] += 1
            answer_lengths.append(len(a_ids) + 1)
            if features is None:
                if len(dropped_examples) < 5:
                    dropped_examples.append(rec["question"][:80])
                continue
            encoded.append(features)

    inference_max_new = get_prompt_spec()["max_new_tokens"]
    over_inference = sum(1 for n in answer_lengths if n > inference_max_new)
    summary = {
        "examples": len(records),
        "kept": len(encoded),
        "truncated": stats.get("truncated", 0),
        "dropped_too_long": stats.get("too_long", 0),
        "dropped_prompt_too_long": stats.get("prompt_too_long", 0),
        "max_answer_tokens": max(answer_lengths) if answer_lengths else 0,
        "answers_longer_than_inference_max_new_tokens": over_inference,
    }
    dropped = summary["dropped_too_long"] + summary["dropped_prompt_too_long"]
    if dropped or summary["truncated"]:
        logger.warning(
            "%s: %d example(s) longer than max_length=%d were %s (policy=%s). Examples: %s",
            name,
            dropped + summary["truncated"],
            max_length,
            "truncated" if truncation_policy == "truncate" else "dropped",
            truncation_policy,
            dropped_examples,
        )
    if over_inference:
        logger.warning(
            "%s: %d answer(s) are longer than the inference max_new_tokens (%d) and will be "
            "cut at generation time",
            name,
            over_inference,
            inference_max_new,
        )
    return encoded, summary


def check_tokenization_consistency(tokenizer, records: Sequence[Dict[str, Any]], limit: int = 50) -> int:
    """Count examples where tokenizing prompt and answer separately differs from
    tokenizing the full text (would mean a train/inference boundary mismatch)."""
    mismatches = 0
    suffix = _appended_special_token_count(tokenizer)
    for rec in list(records)[:limit]:
        prompt = build_prompt(rec["question"], rec["language"])
        separate = _strip_suffix(tokenizer(prompt, add_special_tokens=True, **TOKENIZER_CALL_KWARGS)["input_ids"], suffix) + list(
            tokenizer(ANSWER_PREFIX + rec["answer"], add_special_tokens=False, **TOKENIZER_CALL_KWARGS)["input_ids"]
        )
        joint = _strip_suffix(tokenizer(prompt + ANSWER_PREFIX + rec["answer"], add_special_tokens=True, **TOKENIZER_CALL_KWARGS)["input_ids"], suffix)
        if separate != joint:
            mismatches += 1
    if mismatches:
        logger.warning(
            "%d/%d examples tokenize differently at the prompt/answer boundary", mismatches, min(limit, len(records))
        )
    return mismatches


class CausalLMCollator:
    """Right-pads input_ids/attention_mask/labels. Labels are padded with -100,
    so padding (pad == eos for tokenizers without a pad token) never hides the
    real EOS label. A ``lang`` field (evaluation: 0 Turkish, 1 English) is passed
    through as a tensor."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: Optional[int] = 8, label_pad_token_id: int = IGNORE_INDEX):
        if pad_token_id is None:
            raise ValueError("pad_token_id is required")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    @staticmethod
    def _as_list(value) -> List[int]:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return list(value)

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        rows = [{k: self._as_list(f[k]) for k in ("input_ids", "attention_mask", "labels")} for f in features]
        max_len = max(len(r["input_ids"]) for r in rows)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        input_ids, attention, labels = [], [], []
        for r in rows:
            pad = max_len - len(r["input_ids"])
            input_ids.append(r["input_ids"] + [self.pad_token_id] * pad)
            attention.append(r["attention_mask"] + [0] * pad)
            labels.append(r["labels"] + [self.label_pad_token_id] * pad)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if features and all("lang" in f for f in features):
            batch["lang"] = torch.tensor([int(f["lang"]) for f in features], dtype=torch.long)
        return batch


LANG_IDS = {"turkish": 0, "english": 1}


def to_hf_dataset(encoded: Sequence[Dict[str, List[int]]], langs: Optional[Sequence[int]] = None):
    """HF dataset of encoded examples (+ a ``lang`` column for evaluation)."""
    from datasets import Dataset

    columns: Dict[str, Any] = {
        "input_ids": [e["input_ids"] for e in encoded],
        "attention_mask": [e["attention_mask"] for e in encoded],
        "labels": [e["labels"] for e in encoded],
    }
    if langs is not None:
        columns["lang"] = list(langs)
    return Dataset.from_dict(columns)


def build_eval_dataset(val_tr: Sequence[Dict[str, List[int]]], val_en: Sequence[Dict[str, List[int]]]):
    """One evaluation set for both languages, longest examples first.

    Sorted by length, every eval batch holds similar lengths (about 4% padding
    instead of about 40% in random order); longest first, so an out-of-memory
    shows up in the first batch. The order does not change any metric: the
    losses are token-weighted sums. None if there is no validation data.
    """
    rows = [(e, LANG_IDS["turkish"]) for e in val_tr] + [(e, LANG_IDS["english"]) for e in val_en]
    if not rows:
        return None
    rows.sort(key=lambda r: len(r[0]["input_ids"]), reverse=True)  # stable: deterministic
    return to_hf_dataset([r[0] for r in rows], langs=[r[1] for r in rows])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class SimpleProgressCallback(TrainerCallback):
    """Simple progress logging"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        train_loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        if train_loss is not None:
            logger.info("Step %s/%s - Train Loss: %.4f", state.global_step, state.max_steps, train_loss)
        if eval_loss is not None:
            per_lang = ", ".join(f"{short} {logs[f'eval_loss_{name}']:.4f}"
                                 for name, short in (("turkish", "TR"), ("english", "EN"))
                                 if isinstance(logs.get(f"eval_loss_{name}"), (int, float)))
            logger.info("Step %s/%s - Eval Loss: %.4f%s (%.0fs)", state.global_step, state.max_steps, eval_loss,
                        f" ({per_lang})" if per_lang else "", logs.get("eval_runtime") or 0)


# eval_loss of this script since 2026-11-12 (recorded in training_info.json); older
# runs reported the Trainer's mean of per-batch means, which weighs short batches more.
EVAL_LOSS_DEFINITION = "token-weighted mean over all answer+EOS tokens"


def answer_token_loss_sums(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per example: (sum of the answer-token losses, number of answer tokens).

    Same shift and masking as the causal LM loss of the model; only the label
    positions are up-cast to float32 (not the whole logits tensor).
    """
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    token_losses = torch.nn.functional.cross_entropy(
        logits[:, :-1, :][mask].float(), shift_labels[mask], reduction="none")
    example_of_token = mask.nonzero(as_tuple=True)[0]
    sums = torch.zeros(labels.size(0), dtype=torch.float64, device=logits.device)
    sums.index_add_(0, example_of_token, token_losses.double())
    return sums, mask.sum(dim=1)


class StepSortedRandomSampler(torch.utils.data.Sampler):
    """Random training order with micro-batches of similar length.

    The permutation is the one the Trainer's default sampler draws
    (``torch.randperm`` seeded with ``seed + epoch``, like accelerate's
    ``SeedableRandomSampler``), so every optimizer step (``step_size`` =
    micro-batch x gradient accumulation examples) trains on exactly the same
    examples as without sorting: the step's gradient (summed over its
    micro-batches and normalised by its answer tokens) does not change. Only
    inside a step are the examples ordered by length, longest first, so its
    micro-batches carry little padding (real data, micro-batch 8: 14% instead of
    54%). ``transformers``' ``group_by_length`` was not used: it puts similar
    lengths into the same *step*, which measurably raised the validation loss.
    Deterministic per epoch, so a resumed run replays the same order.
    """

    def __init__(self, lengths: Sequence[int], step_size: int, seed: int = 42):
        self.lengths = [int(n) for n in lengths]
        self.step_size = max(1, int(step_size))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:  # called by the Trainer (via accelerate) every epoch
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        sorted_steps: List[int] = []
        for start in range(0, len(order), self.step_size):
            sorted_steps.extend(sorted(order[start:start + self.step_size], key=lambda i: -self.lengths[i]))
        return iter(sorted_steps)


@lru_cache(maxsize=1)
def cengbot_trainer_class():
    """``transformers.Trainer`` subclass (built lazily: importing Trainer costs ~3 s).

    * One evaluation pass yields ``eval_loss`` and the Turkish/English losses: eval
      batches carry a ``lang`` column and the answer-token loss sums are tallied
      per language inside the Trainer's own loop (no second pass over the data).
      ``eval_loss`` is the token-weighted mean (``EVAL_LOSS_DEFINITION``).
    * ``train_lengths`` set: a ``StepSortedRandomSampler`` for training.
    """
    from transformers import Trainer

    class CengBotTrainer(Trainer):
        train_lengths: Optional[List[int]] = None
        _eval_sums: Optional[torch.Tensor] = None

        def _get_train_sampler(self, train_dataset=None):
            if self.train_lengths is None:
                return super()._get_train_sampler(train_dataset)
            seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
            return StepSortedRandomSampler(
                self.train_lengths, self.args.train_batch_size * self.args.gradient_accumulation_steps, seed)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            lang = inputs.pop("lang", None)
            if self._eval_sums is None or lang is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                            num_items_in_batch=num_items_in_batch)
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            example_sums, example_tokens = answer_token_loss_sums(outputs.logits, labels)
            self._eval_sums[0].index_add_(0, lang, example_sums)
            self._eval_sums[1].index_add_(0, lang, example_tokens.double())
            loss = example_sums.sum() / example_tokens.sum().clamp(min=1)
            return (loss, outputs) if return_outputs else loss

        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                            metric_key_prefix="eval"):
            self._eval_sums = torch.zeros(2, len(LANG_IDS), dtype=torch.float64, device=self.args.device)
            try:
                output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys,
                                                 metric_key_prefix)
                sums = self._eval_sums
                if self.args.world_size > 1:
                    sums = self.accelerator.reduce(sums, reduction="sum")
                loss_sums, tokens = sums.tolist()
            finally:
                self._eval_sums = None
            if sum(tokens):
                output.metrics[f"{metric_key_prefix}_loss"] = sum(loss_sums) / sum(tokens)
                for name, idx in LANG_IDS.items():
                    if tokens[idx]:
                        output.metrics[f"{metric_key_prefix}_loss_{name}"] = loss_sums[idx] / tokens[idx]
            return output

    return CengBotTrainer


# ---------------------------------------------------------------------------
# Data processing facade (kept for backward compatibility)
# ---------------------------------------------------------------------------
class DataProcessor:
    """Handle data loading and processing (thin wrapper around the functions above)."""

    def __init__(self, tokenizer, logger_=None, cfg: Optional[Config] = None):
        self.tokenizer = tokenizer
        self.logger = logger_ or logger
        self.cfg = cfg or config

    def load_dataset(self) -> List[Dict]:
        records, _ = load_training_records(self.cfg.dataset_paths)
        if self.cfg.semantic_dedup:
            records = semantic_deduplicate(records, cache_dir=self.cfg.cache_dir)
        return records

    def deduplicate_dataset(self, data: List[Dict]) -> List[Dict]:
        return clean_records(data)[0]

    def create_balanced_split(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        train, val = split_train_validation(data, self.cfg.val_split_ratio, self.cfg.seed)
        val_tr, val_en = split_by_language(val)
        return train, val_tr, val_en, val

    def format_example(self, example: Dict) -> str:
        return format_training_text(example["question"], example["answer"], example.get("language", "english"))

    def tokenize_and_mask(self, examples):
        """Batched ``datasets.map`` function over a ``raw`` column (drops over-long rows)."""
        encoded, _ = tokenize_records(
            [
                {"question": r["question"], "answer": r["answer"], "language": normalize_language(r.get("language")) or "english"}
                for r in examples["raw"]
            ],
            self.tokenizer,
            self.cfg.max_length,
            self.cfg.truncation_policy,
        )
        return {
            "input_ids": [e["input_ids"] for e in encoded],
            "attention_mask": [e["attention_mask"] for e in encoded],
            "labels": [e["labels"] for e in encoded],
        }


# ---------------------------------------------------------------------------
# Model loading (offline-first)
# ---------------------------------------------------------------------------
def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _is_offline() -> bool:
    return _truthy(os.getenv("HF_HUB_OFFLINE")) or _truthy(os.getenv("TRANSFORMERS_OFFLINE"))


def downloads_allowed() -> bool:
    """Same policy as inference (llama_model_handler.downloads_allowed): no downloads in
    HF offline mode or with MODEL_ALLOW_DOWNLOAD=false."""
    if _is_offline():
        return False
    return str(os.getenv("MODEL_ALLOW_DOWNLOAD", "")).strip().lower() not in ("0", "false", "no", "off", "disabled")


def get_hf_token() -> Optional[str]:
    for var in ("HUGGING_FACE_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.getenv(var, "").strip()
        lowered = value.lower()
        if value and not lowered.startswith(("your_", "your-", "<")) and "placeholder" not in lowered:
            return value
    return None


def is_model_cached(model_name: str, cache_dir: Optional[str]) -> bool:
    """True if config, tokenizer and *all* weight files of ``model_name`` are in ``cache_dir``."""
    if os.path.isdir(model_name):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return False

    def cached(filename: str) -> Optional[str]:
        try:
            path = try_to_load_from_cache(model_name, filename, cache_dir=cache_dir)
        except Exception:
            return None
        return path if isinstance(path, str) and os.path.isfile(path) else None

    if not cached("config.json"):
        return False
    if not (cached("tokenizer.json") or cached("tokenizer_config.json")):
        return False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = cached(index_name)
        if index_path:
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    shards = set(json.load(f).get("weight_map", {}).values())
            except Exception:
                return False
            return bool(shards) and all(cached(shard) for shard in shards)
    return bool(cached("model.safetensors") or cached("pytorch_model.bin"))


def resolve_model_source(model_name: str, cache_dir: Optional[str]) -> Dict[str, Any]:
    """``from_pretrained`` kwargs: local cache when available, network only as fallback."""
    if os.path.isdir(model_name):
        return {"local_files_only": True}
    if is_model_cached(model_name, cache_dir):
        return {"cache_dir": cache_dir, "local_files_only": True}
    if not downloads_allowed():
        raise RuntimeError(
            f"Base model '{model_name}' is not fully cached in '{cache_dir}' and downloads are disabled "
            "(HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE or MODEL_ALLOW_DOWNLOAD=false). Download it first "
            "(or allow the download once) - see docs/MODEL_SWAP.md."
        )
    token = get_hf_token()
    if token is None:
        logger.warning(
            "Base model '%s' is not cached and HUGGING_FACE_TOKEN is not set - the download "
            "will fail for gated models (e.g. meta-llama/*, google/gemma-*, some mistralai/*: accept "
            "the license on huggingface.co and set HUGGING_FACE_TOKEN)",
            model_name,
        )
    logger.warning("Base model '%s' not cached - downloading into %s", model_name, cache_dir)
    source: Dict[str, Any] = {"cache_dir": cache_dir, "local_files_only": False}
    if token:
        source["token"] = token
    return source


def detect_precision(force_cpu: bool = False) -> Dict[str, Any]:
    cuda = torch.cuda.is_available() and not force_cpu
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    fp16 = bool(cuda and not bf16)
    tf32 = False
    if cuda:
        try:
            tf32 = torch.cuda.get_device_capability()[0] >= 8
        except Exception:
            tf32 = False
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    return {"cuda": cuda, "bf16": bf16, "fp16": fp16, "tf32": tf32, "dtype": dtype}


def _remote_code_kwargs(cfg: Config) -> Dict[str, Any]:
    # Only passed when enabled, so default runs never mention trust_remote_code.
    return {"trust_remote_code": True} if cfg.trust_remote_code else {}


def _generation_config_eos(cfg: Config, source: Dict[str, Any]) -> Optional[int]:
    try:
        from transformers import GenerationConfig

        eos = GenerationConfig.from_pretrained(cfg.model_name, **source).eos_token_id
    except Exception:
        return None
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    return int(eos) if isinstance(eos, int) else None


def load_tokenizer(cfg: Optional[Config] = None):
    from transformers import AutoTokenizer

    cfg = cfg or config
    source = resolve_model_source(cfg.model_name, cfg.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, **source, **_remote_code_kwargs(cfg))
    if getattr(tokenizer, "eos_token_id", None) is None:
        eos = _generation_config_eos(cfg, source)
        if eos is not None:
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(eos)
            logger.warning("Tokenizer has no EOS token - using %r (id %d) from generation_config.json",
                           tokenizer.eos_token, eos)
    _eos_id(tokenizer)
    if tokenizer.pad_token is None:  # no pad token (Llama, Mistral, Qwen base ...): pad with EOS
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # training; saved as "left" for inference
    return tokenizer


def _model_type(cfg: Config, source: Dict[str, Any]) -> Optional[str]:
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(cfg.model_name, **source, **_remote_code_kwargs(cfg)).model_type
    except Exception:  # from_pretrained of the model reports the real problem
        return None


def load_base_model(cfg: Optional[Config] = None, precision: Optional[Dict[str, Any]] = None):
    from transformers import AutoModelForCausalLM

    cfg = cfg or config
    precision = precision or detect_precision(cfg.force_cpu)
    source = resolve_model_source(cfg.model_name, cfg.cache_dir)
    kwargs: Dict[str, Any] = {"torch_dtype": precision["dtype"], **source, **_remote_code_kwargs(cfg)}
    model_type = _model_type(cfg, source)
    if model_type in EAGER_ATTENTION_MODEL_TYPES:
        kwargs["attn_implementation"] = "eager"
        logger.info("%s model: training with eager attention (recommended by transformers)", model_type)
    if cfg.load_in_4bit:
        if not precision["cuda"]:
            raise RuntimeError("--load-in-4bit requires a CUDA GPU")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=precision["dtype"],
        )
    if precision["cuda"]:
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing
    logger.info(
        "Loaded base model %s (architecture=%s, dtype=%s, 4bit=%s, attention=%s, params=%.2fB)",
        cfg.model_name,
        getattr(model.config, "model_type", "?"),
        precision["dtype"],
        cfg.load_in_4bit,
        getattr(model.config, "_attn_implementation", "?"),
        sum(p.numel() for p in model.parameters()) / 1e9,
    )
    return model


def parse_lora_target_modules(value: Any) -> Any:
    """``"auto"`` | ``"all-linear"`` | ``"q_proj,v_proj"`` (or a list) -> "auto" | "all-linear" | [names]."""
    if value is None:
        return LORA_TARGET_AUTO
    if isinstance(value, (list, tuple, set)):
        names = [str(v).strip() for v in value if str(v).strip()]
    else:
        text = str(value).strip()
        if not text or text.lower() == LORA_TARGET_AUTO:
            return LORA_TARGET_AUTO
        if text.lower() in (LORA_TARGET_ALL_LINEAR, "all_linear"):
            return LORA_TARGET_ALL_LINEAR
        names = [n.strip() for n in text.split(",") if n.strip()]
    if not names:
        raise ValueError(f"Invalid LoRA target modules {value!r}: use auto, all-linear or a comma separated list")
    return names


def linear_module_names(model) -> set:
    """Leaf names of the Linear layers LoRA can adapt (the output head excluded)."""
    try:
        from transformers.pytorch_utils import Conv1D
        linear_types: Tuple[type, ...] = (torch.nn.Linear, Conv1D)
    except ImportError:  # pragma: no cover
        linear_types = (torch.nn.Linear,)
    try:
        output_head = model.get_output_embeddings()
    except Exception:
        output_head = None
    return {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if name and module is not output_head and isinstance(module, linear_types)  # bnb Linear4bit is a Linear
    }


def resolve_lora_target_modules(model, spec: Any) -> Tuple[Any, str]:
    """Return ``(target_modules for LoraConfig, mode)``; mode is explicit | all-linear | custom.

    ``auto``: the explicit projection list if the model has every one of those
    Linear layers (Llama, Qwen, Mistral, Gemma, Granite ...), else ``all-linear``
    (fused projections: Phi-3/4 ``qkv_proj``/``gate_up_proj``, GLM ``gate_up_proj``).
    PEFT's all-linear never adapts the output head (lm_head).
    """
    spec = parse_lora_target_modules(spec)
    if spec == LORA_TARGET_ALL_LINEAR:
        return LORA_TARGET_ALL_LINEAR, "all-linear"
    available = linear_module_names(model)
    if spec == LORA_TARGET_AUTO:
        if set(LORA_TARGET_MODULES) <= available:
            return list(LORA_TARGET_MODULES), "explicit"
        logger.info("LoRA targets (auto): the model has no %s layers (Linear layers: %s) - using all-linear",
                    "/".join(sorted(set(LORA_TARGET_MODULES) - available)), ", ".join(sorted(available)))
        return LORA_TARGET_ALL_LINEAR, "all-linear"
    missing = [name for name in spec if name not in available]
    if len(missing) == len(spec):
        raise ValueError(f"None of the LoRA target modules {spec} exist in this model "
                         f"(Linear layers: {', '.join(sorted(available))})")
    if missing:
        logger.warning("LoRA target modules not found in this model and ignored: %s", ", ".join(missing))
    return list(spec), "custom"


def lora_target_summary(model) -> List[str]:
    """Sorted leaf names of the modules a PeftModel actually adapts."""
    try:
        targets = model.peft_config["default"].target_modules
    except Exception:
        return []
    if isinstance(targets, str):
        return [targets]
    return sorted({str(t).rsplit(".", 1)[-1] for t in targets})


def apply_lora(model, cfg: Optional[Config] = None, target_modules: Any = None):
    """Wrap the base model with LoRA adapters (QLoRA prep only for 4-bit models).

    ``target_modules`` defaults to ``resolve_lora_target_modules(model, cfg.lora_target_modules)``.
    """
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    cfg = cfg or config
    if target_modules is None:
        target_modules, _ = resolve_lora_target_modules(model, cfg.lora_target_modules)
    if cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cfg.gradient_checkpointing:
        # prepare_model_for_kbit_training would up-cast the whole bf16 model to fp32.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules if isinstance(target_modules, str) else list(target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    try:
        model.print_trainable_parameters()
    except Exception:  # pragma: no cover
        pass
    return model


def setup_model_and_tokenizer(cfg: Optional[Config] = None):
    """Load tokenizer + base model (offline-first) and apply LoRA."""
    cfg = cfg or config
    tokenizer = load_tokenizer(cfg)
    model = apply_lora(load_base_model(cfg), cfg)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def plan_training_steps(cfg: Config, n_train: int) -> Dict[str, Any]:
    """Optimizer steps per epoch / in total and the eval (= checkpoint) interval.

    Same arithmetic as the HF Trainer. ``eval_steps`` spreads ``evals_per_epoch``
    evaluations over one epoch, or over the whole run when it is shorter
    (``--max-steps`` / fractional ``--epochs``), so short runs still evaluate and
    write checkpoints.
    """
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, math.ceil(n_train / effective_batch))
    if cfg.max_steps and cfg.max_steps > 0:
        total_steps = int(cfg.max_steps)
    else:
        total_steps = max(1, math.ceil(cfg.num_epochs * steps_per_epoch))
    eval_steps = max(1, min(steps_per_epoch, total_steps) // max(1, cfg.evals_per_epoch))
    return {
        "train_examples": n_train,
        "effective_batch_size": effective_batch,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "epochs": round(total_steps / steps_per_epoch, 3),
        "eval_and_save_every_steps": eval_steps,
    }


def build_training_arguments(cfg: Config, run_dir: str, n_train: int, has_eval: bool, precision: Dict[str, Any]):
    from transformers import TrainingArguments

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    eval_steps = plan_training_steps(cfg, n_train)["eval_and_save_every_steps"]
    workers = cfg.dataloader_num_workers if precision["cuda"] else 0
    kwargs: Dict[str, Any] = dict(
        output_dir=run_dir,
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else -1,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=cfg.save_total_limit,
        save_safetensors=True,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        dataloader_num_workers=workers,
        dataloader_pin_memory=precision["cuda"],
        remove_unused_columns=False,
        report_to=[],
        run_name=f"cengbot-{datetime.now().strftime('%Y%m%d-%H%M')}",
        label_names=["labels"],  # PeftModel hides forward() args; without this eval_loss can go missing
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if precision["cuda"] else "adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        seed=cfg.seed,
        data_seed=cfg.seed,
        use_cpu=not precision["cuda"],
        # --resume-from-checkpoint also restores the early-stopping patience counter
        # (saved in every checkpoint, ignored by the Trainer by default).
        restore_callback_states_from_checkpoint=True,
    )
    if precision["tf32"]:
        kwargs["tf32"] = True
    if workers > 0:
        kwargs.update(dataloader_persistent_workers=True, dataloader_prefetch_factor=2)
    if world_size > 1:
        kwargs["ddp_find_unused_parameters"] = False
    if has_eval:
        kwargs.update(
            eval_strategy="steps",
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        kwargs.update(eval_strategy="no", load_best_model_at_end=False)
    return TrainingArguments(**kwargs)


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    cfg: Optional[Config] = None,
    run_dir: Optional[str] = None,
    precision: Optional[Dict[str, Any]] = None,
):
    """Create the Trainer (answer-only loss collator, early stopping, one eval pass with
    TR/EN losses: ``eval_dataset`` from ``build_eval_dataset``, with a ``lang`` column)."""
    from transformers import EarlyStoppingCallback, Trainer

    cfg = cfg or config
    precision = precision or detect_precision(cfg.force_cpu)
    run_dir = run_dir or str(Path(cfg.checkpoint_root) / (cfg.output_name or "run"))
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    args = build_training_arguments(cfg, run_dir, len(train_dataset), has_eval, precision)
    collator = CausalLMCollator(tokenizer.pad_token_id)

    callbacks: List[TrainerCallback] = [SimpleProgressCallback()]
    if has_eval:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        data_collator=collator,
        callbacks=callbacks,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # pragma: no cover - transformers < 4.46
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = cengbot_trainer_class()(**trainer_kwargs)
    if cfg.sort_by_length:
        trainer.train_lengths = [len(ids) for ids in train_dataset["input_ids"]]
    return trainer


# ---------------------------------------------------------------------------
# Versioning, saving, logging
# ---------------------------------------------------------------------------
def get_next_version_number(history_dir: Optional[str] = None) -> int:
    """Next training-session number (logs/training_history/v<N>.log)."""
    history_dir = history_dir or config.history_dir
    os.makedirs(history_dir, exist_ok=True)
    versions = []
    for name in os.listdir(history_dir):
        m = re.fullmatch(r"v(\d+)\.log", name)
        if m:
            versions.append(int(m.group(1)))
    return max(versions) + 1 if versions else 1


def get_next_model_version(models_dir: Optional[str] = None) -> int:
    """Next integer model version: v1, v1.1, v1.2 -> 2."""
    models_dir = models_dir or config.output_dir
    if not os.path.isdir(models_dir):
        return 1
    majors = []
    for name in os.listdir(models_dir):
        if name.startswith(MODEL_DIR_PREFIX):
            m = re.fullmatch(r"(\d+)(?:\.\d+)*", name[len(MODEL_DIR_PREFIX):])
            if m:
                majors.append(int(m.group(1)))
    return max(majors) + 1 if majors else 1


_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_final_model_dir(models_dir: str, name: str, overwrite: bool = False) -> Path:
    """Validate the output name and make sure we will not clobber a model."""
    if not _VALID_NAME_RE.match(name) or name == ACTIVE_MODEL_LINK:
        raise ValueError(f"Invalid output model name: {name!r}")
    final_dir = Path(models_dir) / name
    active = Path(models_dir) / ACTIVE_MODEL_LINK
    if final_dir.exists() or final_dir.is_symlink():
        if not overwrite:
            raise FileExistsError(f"Model directory already exists: {final_dir} (use --overwrite or another --output-name)")
        if final_dir.is_symlink() or (active.exists() and final_dir.resolve() == active.resolve()):
            raise ValueError(f"Refusing to overwrite the active model or a symlink: {final_dir}")
    return final_dir


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
_CHECKPOINT_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "pytorch_model.bin")


def checkpoint_problem(path: Path) -> Optional[str]:
    """None if ``path`` is a complete Trainer checkpoint, else what is missing.

    The Trainer writes a checkpoint in place and ``trainer_state.json`` last, so a
    run killed while saving leaves a directory that cannot be resumed from.
    """
    path = Path(path)
    if not path.is_dir():
        return "not a directory"
    if not any((path / name).is_file() for name in _CHECKPOINT_WEIGHTS):
        return "no adapter/model weights"
    if not (path / "optimizer.pt").is_file():
        return "no optimizer.pt"
    try:
        with open(path / "trainer_state.json", "r", encoding="utf-8") as f:
            json.load(f)
    except (OSError, ValueError):
        return "no readable trainer_state.json"
    return None


def find_resumable_checkpoint(run_dir: str) -> Optional[str]:
    """Newest complete ``checkpoint-<step>`` in ``run_dir`` (incomplete ones are skipped)."""
    run = Path(run_dir)
    if not run.is_dir():
        return None
    candidates = [p for p in run.iterdir() if p.is_dir() and _CHECKPOINT_RE.fullmatch(p.name)]
    for path in sorted(candidates, key=lambda p: int(_CHECKPOINT_RE.fullmatch(p.name).group(1)), reverse=True):
        problem = checkpoint_problem(path)
        if problem is None:
            return str(path)
        logger.warning("Skipping incomplete checkpoint %s (%s - the run was probably stopped while saving it)",
                       path, problem)
    return None


def checkpoint_global_step(path: str) -> int:
    """``global_step`` recorded in a checkpoint's trainer_state.json (0 if unknown)."""
    try:
        with open(Path(path) / "trainer_state.json", "r", encoding="utf-8") as f:
            return int(json.load(f).get("global_step") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def whole_run_train_loss(train_loss: Optional[float], global_step: int, resumed_at_step: int,
                         log_history: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Mean train loss of the whole run, also after ``--resume-from-checkpoint``.

    After a resume the Trainer divides the loss summed over the resumed steps only
    by *all* steps, so its ``train_loss`` is too low by ``resumed_at_step/global_step``.
    The steps before the resume are taken from the logged loss windows the
    checkpoint carried over (``log_history``).
    """
    if train_loss is None or not resumed_at_step or global_step <= resumed_at_step:
        return train_loss
    session_sum = float(train_loss) * global_step
    windows, last = [], 0
    logged = [e for e in log_history if "loss" in e and isinstance(e.get("step"), int) and e["step"] <= resumed_at_step]
    for entry in sorted(logged, key=lambda e: e["step"]):
        if entry["step"] > last:
            windows.append((float(entry["loss"]), entry["step"] - last))
            last = entry["step"]
    weight = sum(w for _, w in windows)
    if not weight:  # nothing logged before the resume: mean over the resumed steps
        return session_sum / (global_step - resumed_at_step)
    before_mean = sum(loss * w for loss, w in windows) / weight
    return (before_mean * resumed_at_step + session_sum) / global_step


def final_eval_metrics(state) -> Dict[str, Any]:
    """The logged evaluation of the checkpoint the run ends with: the best one
    (``load_best_model_at_end``). Empty if it cannot be identified."""
    best = getattr(state, "best_model_checkpoint", None)
    match = _CHECKPOINT_RE.fullmatch(Path(best).name) if best else None
    if match is None:
        return {}
    step = int(match.group(1))
    logged = [entry for entry in state.log_history if "eval_loss" in entry and entry.get("step") == step]
    return dict(logged[-1]) if logged else {}


DATA_FINGERPRINT_FILE = "data_fingerprint.json"


def data_fingerprint(data: Dict[str, Any]) -> Dict[str, Any]:
    """SHA-256 of the tokenized train/validation split (ids + labels, in order).

    Written next to the checkpoints of a run; a resumed run must see exactly the
    same examples, otherwise the Trainer would skip batches of a different data
    order and validation examples could leak into training.
    """
    digest = hashlib.sha256()
    counts: Dict[str, int] = {}
    for part in ("train", "val_tr", "val_en"):
        rows = data.get(part) or []
        counts[part] = len(rows)
        digest.update(f"{part}:{len(rows)};".encode())
        for row in rows:
            ids, labels = row["input_ids"], row["labels"]
            digest.update(len(ids).to_bytes(4, "little"))
            digest.update(array("q", ids).tobytes())
            digest.update(array("q", labels).tobytes())
    return {"sha256": digest.hexdigest(), "train_examples": counts["train"],
            "val_examples_turkish": counts["val_tr"], "val_examples_english": counts["val_en"]}


def check_resume_data(checkpoint: str, fingerprint: Dict[str, Any]) -> None:
    """Raise if the data differ from the run that wrote ``checkpoint`` (its
    ``data_fingerprint.json`` lives next to it). Older runs without one: warning only."""
    path = Path(checkpoint).parent / DATA_FINGERPRINT_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        logger.warning("No readable %s next to %s - cannot verify that the resumed run sees the same data",
                       DATA_FINGERPRINT_FILE, checkpoint)
        return
    if saved.get("sha256") == fingerprint["sha256"]:
        return
    counts = ", ".join(f"{k} {saved.get(k)} -> {fingerprint[k]}" for k in fingerprint if k != "sha256")
    raise ValueError(
        f"--resume-from-checkpoint {checkpoint}: the training data differ from the interrupted run ({counts}). "
        "Resuming would continue on another train/validation split and data order. Use the same dataset files "
        "and --val-ratio/--seed/--max-length/--truncation/--semantic-dedup as that run (the database export is "
        "skipped on resume), or start a new run without --resume-from-checkpoint."
    )


def resolve_resume_checkpoint(value: Optional[str], run_dir: str) -> Optional[str]:
    """``--resume-from-checkpoint`` -> checkpoint directory (``latest``: newest complete
    one in ``run_dir``). Raises before the model is loaded if there is nothing to resume."""
    if not value:
        return None
    if value == "latest":
        found = find_resumable_checkpoint(run_dir)
        if found is None:
            raise FileNotFoundError(
                f"--resume-from-checkpoint latest: no complete checkpoint in {run_dir}. Pass the same "
                "--output-name/--checkpoint-dir as the interrupted run (its checkpoints live in "
                "<checkpoint-dir>/<output-name>/checkpoint-<step>) or a checkpoint directory."
            )
        return found
    path = Path(value).expanduser()
    problem = checkpoint_problem(path)
    if problem is not None:
        raise FileNotFoundError(f"--resume-from-checkpoint {value}: not a usable checkpoint ({problem})")
    return str(path)


def _write_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_NON_MODEL_FILES = ("optimizer.pt", "scheduler.pt", "training_args.bin", "trainer_state.json", "rng_state.pth")


def save_final_model(model, tokenizer, final_dir: Path, training_info: Dict[str, Any], overwrite: bool = False) -> Path:
    """Save the adapter + tokenizer + training_info.json atomically into ``final_dir``.

    The model is written to a hidden temp dir next to ``final_dir`` and renamed
    into place, so a crash never leaves a half-written model that
    ``switch_model.sh`` could activate.
    """
    final_dir = Path(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = final_dir.parent / f".{final_dir.name}.tmp-{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    try:
        try:
            model.save_pretrained(str(tmp_dir), safe_serialization=True, save_embedding_layers=False)
        except TypeError:
            model.save_pretrained(str(tmp_dir), safe_serialization=True)
        missing = [f for f in ("adapter_config.json", "adapter_model.safetensors") if not (tmp_dir / f).is_file()]
        if missing:
            raise RuntimeError(f"Adapter save incomplete, missing: {missing}")
        adapter_cfg_path = tmp_dir / "adapter_config.json"
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        if not adapter_cfg.get("base_model_name_or_path") and training_info.get("base_model"):
            adapter_cfg["base_model_name_or_path"] = training_info["base_model"]
            _write_json_atomic(adapter_cfg_path, adapter_cfg)

        old_side = getattr(tokenizer, "padding_side", None)
        try:
            tokenizer.padding_side = "left"  # inference convention
            tokenizer.save_pretrained(str(tmp_dir))
        finally:
            if old_side is not None:
                tokenizer.padding_side = old_side
        for name in _NON_MODEL_FILES:
            (tmp_dir / name).unlink(missing_ok=True)
        _write_json_atomic(tmp_dir / "training_info.json", training_info)

        if final_dir.exists():
            if not overwrite or final_dir.is_symlink():
                raise FileExistsError(f"Model directory already exists: {final_dir}")
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    logger.info("Final model saved to %s", final_dir)
    return final_dir


def write_metrics_csv(log_history: Sequence[Dict[str, Any]], path: Path) -> None:
    """Write Trainer log history; rows have different keys, so use their union."""
    if not log_history:
        return
    fieldnames: List[str] = []
    for row in log_history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(log_history)


def setup_logging(cfg: Optional[Config] = None, version_number: Optional[int] = None, to_files: bool = True):
    """Console logging always; versioned + timestamped log files for real runs."""
    cfg = cfg or config
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_files: Dict[str, str] = {}
    if to_files:
        os.makedirs(cfg.history_dir, exist_ok=True)
        if version_number is None:
            version_number = get_next_version_number(cfg.history_dir)
        versioned = Path(cfg.history_dir) / f"v{version_number}.log"
        timestamped = Path(cfg.log_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        for path in (versioned, timestamped):
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        log_files = {"versioned": str(versioned), "timestamped": str(timestamped)}
        logger.info("=== CengBot Training Session v%s ===", version_number)
        logger.info("Versioned log: %s", versioned)
    return logger, log_files


def _resolve_history_version(cfg: Config) -> int:
    env_version = os.environ.get("TRAINING_VERSION", "").strip()
    if env_version.isdigit() and int(env_version) > 0:
        return int(env_version)
    return get_next_version_number(cfg.history_dir)


def _load_env() -> None:
    """Load the project .env (never overrides variables already set)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def run_database_export(cfg: Config) -> Optional[Dict[str, Any]]:
    """Refresh data/cengbot_qa_augmented.jsonl from approved DB rows (non-fatal)."""
    export_file = str(Path(cfg.base_path) / "data" / DB_EXPORT_FILENAME)
    if export_file not in [str(Path(p)) for p in cfg.dataset_paths]:
        logger.info("DB export skipped: %s is not one of the training datasets", export_file)
        return None
    try:
        from database_to_training import DatabaseToTrainingConverter

        result = DatabaseToTrainingConverter(cfg.base_path).full_export(create_backup=True)
    except Exception as exc:
        logger.warning("Database export failed (%s) - continuing with existing files", exc)
        return None
    if result.get("success"):
        logger.info("Database export: %s", result.get("message"))
    else:
        logger.info("Database export: %s", result.get("message"))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    d = Config()
    p = argparse.ArgumentParser(description="Fine-tune the CengBot LoRA adapter on a causal LM base model "
                                             f"(default {DEFAULT_BASE_MODEL}).")
    p.add_argument("--dataset", nargs="+", default=None,
                   help="JSONL dataset file(s) (default: env TRAINING_DATASET or "
                        "data/merged_qa_data.jsonl + data/cengbot_qa_augmented.jsonl)")
    p.add_argument("--base-model", default=None, help="Base model id/path (default: env BASE_MODEL_NAME or %s)" % DEFAULT_BASE_MODEL)
    p.add_argument("--cache-dir", default=_default_cache_dir(),
                   help="HF cache dir for the base model (default: env MODEL_CACHE_DIR or model_cache/)")
    p.add_argument("--models-dir", default=d.output_dir, help="Where the final model dir is created (default: models/)")
    p.add_argument("--output-name", default=None, help="Final model dir name (default: final-best-model-v<next>)")
    p.add_argument("--checkpoint-dir", default=d.checkpoint_root, help="Root for Trainer checkpoints (default: checkpoints/)")
    p.add_argument("--log-dir", default=d.log_dir, help="Log directory (default: logs/)")
    p.add_argument("--epochs", type=float, default=d.num_epochs)
    p.add_argument("--max-steps", type=int, default=d.max_steps,
                   help="Stop after N optimizer steps (overrides --epochs; for smoke tests). Default: off")
    p.add_argument("--logging-steps", type=int, default=d.logging_steps, help="Log the train loss every N steps")
    p.add_argument("--batch-size", type=int, default=d.per_device_train_batch_size)
    p.add_argument("--eval-batch-size", type=int, default=d.per_device_eval_batch_size)
    p.add_argument("--grad-accum", type=int, default=d.gradient_accumulation_steps)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--truncation", choices=["drop", "truncate"], default=d.truncation_policy,
                   help="What to do with examples longer than --max-length (always reported)")
    p.add_argument("--val-ratio", type=float, default=d.val_split_ratio)
    p.add_argument("--evals-per-epoch", type=int, default=d.evals_per_epoch)
    p.add_argument("--early-stopping-patience", type=int, default=d.early_stopping_patience)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--lora-r", type=int, default=d.lora_r)
    p.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    p.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    p.add_argument("--lora-target-modules", default=None, type=_lora_targets_arg,
                   help="auto (default; env LORA_TARGET_MODULES): q/k/v/o/gate/up/down_proj if the model has "
                        "them all, else all-linear | all-linear | comma separated module names")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="QLoRA (4-bit NF4 base; needed for 7-12B models on a 24 GB GPU). Default: bf16 base like inference")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Execute Python code from the model repository (env MODEL_TRUST_REMOTE_CODE). SECURITY RISK - "
                        "only for legacy remote-code models (chatglm3, internlm2) from repositories you trust")
    p.add_argument("--cpu", action="store_true", help="Force CPU (debugging only)")
    p.add_argument("--workers", type=int, default=d.dataloader_num_workers,
                   help=f"DataLoader worker processes (GPU runs; default {d.dataloader_num_workers})")
    p.add_argument("--sort-by-length", action=argparse.BooleanOptionalAction, default=None,
                   help="Order the examples of each optimizer step by length: micro-batches with little "
                        "padding, the same examples per step as random order (env TRAINING_SORT_BY_LENGTH). "
                        f"Default: {'on' if d.sort_by_length else 'off'}")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None,
                   help="Recompute activations in the backward pass: less GPU memory, slower "
                        f"(env TRAINING_GRADIENT_CHECKPOINTING). Default: {'on' if d.gradient_checkpointing else 'off'}")
    p.add_argument("--semantic-dedup", action="store_true", help="Also remove near-duplicates (sentence-transformers)")
    p.add_argument("--no-db-export", action="store_true", help="Do not refresh data/cengbot_qa_augmented.jsonl from the DB")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume from a checkpoint dir, or 'latest' (newest complete one of this --output-name); "
                        "skips the DB export and stops if the training data changed")
    p.add_argument("--overwrite", action="store_true", help="Overwrite an existing (non-active) model dir")
    p.add_argument("--dry-run", action="store_true",
                   help="Load, clean, split and tokenize the data and print stats; no model, no writes")
    return p.parse_args(argv)


def _default_cache_dir() -> str:
    value = os.getenv("MODEL_CACHE_DIR", "").strip()
    if not value:
        return str(PROJECT_ROOT / "model_cache")
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _lora_targets_arg(value: str) -> Any:
    try:
        return parse_lora_target_modules(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _flag_or_env(value: Optional[bool], env_name: str, default: bool) -> bool:
    """CLI --x/--no-x wins, then env (true/false/1/0/...), then the default."""
    if value is not None:
        return bool(value)
    raw = os.getenv(env_name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def build_config(args: argparse.Namespace) -> Config:
    if args.dataset:
        datasets = [str(Path(p).expanduser().resolve()) for p in args.dataset]
    elif os.getenv("TRAINING_DATASET"):
        datasets = [
            str((PROJECT_ROOT / p.strip()).resolve()) if not os.path.isabs(p.strip()) else p.strip()
            for p in re.split(r"[,:;]", os.environ["TRAINING_DATASET"])
            if p.strip()
        ]
    else:
        datasets = [str(PROJECT_ROOT / p) for p in DEFAULT_DATASET_FILES]
    return Config(
        dataset_paths=datasets,
        output_dir=str(Path(args.models_dir).expanduser()),
        checkpoint_root=str(Path(args.checkpoint_dir).expanduser()),
        log_dir=str(Path(args.log_dir).expanduser()),
        cache_dir=str(Path(args.cache_dir).expanduser()),
        output_name=args.output_name,
        model_name=args.base_model or os.getenv("BASE_MODEL_NAME", "").strip() or DEFAULT_BASE_MODEL,
        num_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        logging_steps=max(1, args.logging_steps),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        truncation_policy=args.truncation,
        val_split_ratio=args.val_ratio,
        evals_per_epoch=args.evals_per_epoch,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=args.load_in_4bit,
        lora_target_modules=(args.lora_target_modules if args.lora_target_modules is not None
                             else parse_lora_target_modules(os.getenv("LORA_TARGET_MODULES"))),
        trust_remote_code=bool(args.trust_remote_code) or _truthy(os.getenv("MODEL_TRUST_REMOTE_CODE")),
        force_cpu=args.cpu,
        dataloader_num_workers=max(0, args.workers),
        sort_by_length=_flag_or_env(args.sort_by_length, "TRAINING_SORT_BY_LENGTH", Config.sort_by_length),
        gradient_checkpointing=_flag_or_env(args.gradient_checkpointing, "TRAINING_GRADIENT_CHECKPOINTING",
                                            Config.gradient_checkpointing),
        semantic_dedup=args.semantic_dedup,
        export_from_db=not args.no_db_export,
        resume_from_checkpoint=args.resume_from_checkpoint,
        overwrite_output=args.overwrite,
    )


def _library_versions() -> Dict[str, str]:
    versions = {"torch": torch.__version__}
    for mod in ("transformers", "peft", "datasets"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            pass
    return versions


def prepare_data(cfg: Config, tokenizer) -> Dict[str, Any]:
    """Load -> clean/dedup -> group split -> tokenize. Returns everything training needs."""
    records, load_stats = load_training_records(cfg.dataset_paths)
    if cfg.semantic_dedup:
        records = semantic_deduplicate(records, cache_dir=cfg.cache_dir)
    if not records:
        raise ValueError(f"No usable training examples in {cfg.dataset_paths}")
    groups = assign_groups(records)
    train_records, val_records = split_train_validation(records, cfg.val_split_ratio, cfg.seed, groups)
    val_tr, val_en = split_by_language(val_records)
    logger.info(
        "Examples: %d (groups: %d) -> train %d, val %d (TR %d / EN %d)",
        len(records), len(set(groups)), len(train_records), len(val_records), len(val_tr), len(val_en),
    )
    mismatches = check_tokenization_consistency(tokenizer, records)
    train_enc, train_stats = tokenize_records(train_records, tokenizer, cfg.max_length, cfg.truncation_policy, name="train")
    val_tr_enc, val_tr_stats = tokenize_records(val_tr, tokenizer, cfg.max_length, cfg.truncation_policy, name="val_tr")
    val_en_enc, val_en_stats = tokenize_records(val_en, tokenizer, cfg.max_length, cfg.truncation_policy, name="val_en")
    if not train_enc:
        raise ValueError("No training examples left after tokenization (check --max-length)")
    if len(train_enc) < 100:
        logger.warning("Only %d training examples - the model will barely change", len(train_enc))
    lang_counts = Counter(r["language"] for r in train_records)
    summary = {
        "dataset": load_stats,
        "examples_after_dedup": len(records),
        "groups": len(set(groups)),
        "train_examples": len(train_enc),
        "train_by_language": dict(lang_counts),
        "val_examples": len(val_tr_enc) + len(val_en_enc),
        "val_examples_turkish": len(val_tr_enc),
        "val_examples_english": len(val_en_enc),
        "tokenization": {"train": train_stats, "val_turkish": val_tr_stats, "val_english": val_en_stats,
                         "boundary_mismatches": mismatches},
        "max_length": cfg.max_length,
        "prompt_format": prompt_format_info(),
    }
    return {
        "train": train_enc,
        "val_tr": val_tr_enc,
        "val_en": val_en_enc,
        "summary": summary,
    }


def _run(cfg: Config, dry_run: bool, history_version: Optional[int], log_files: Dict[str, str]) -> int:
    started = time.time()
    if cfg.trust_remote_code:
        logger.warning(
            "SECURITY: --trust-remote-code / MODEL_TRUST_REMOTE_CODE is enabled - Python code from the "
            "repository of %s will be EXECUTED. Only use it for legacy remote-code models (chatglm3, "
            "internlm2) from repositories you trust; inference needs MODEL_TRUST_REMOTE_CODE=true too.",
            cfg.model_name,
        )
    if cfg.export_from_db and not dry_run:
        if cfg.resume_from_checkpoint:
            # A resumed run must see the data it started with (checked below).
            logger.info("Resuming: database export skipped - the run continues on its original data")
        else:
            run_database_export(cfg)

    tokenizer = load_tokenizer(cfg)
    data = prepare_data(cfg, tokenizer)
    summary = data["summary"]
    plan = plan_training_steps(cfg, summary["train_examples"])
    logger.info("Plan: %d optimizer steps (%d per epoch, effective batch %d), eval + checkpoint every %d steps",
                plan["total_steps"], plan["steps_per_epoch"], plan["effective_batch_size"],
                plan["eval_and_save_every_steps"])
    if dry_run:
        print(json.dumps({**summary, "plan": plan}, ensure_ascii=False, indent=2, default=str))
        logger.info("Dry run finished - no model loaded, nothing written")
        return 0

    model_version = get_next_model_version(cfg.output_dir)
    name = cfg.output_name or f"{MODEL_DIR_PREFIX}{model_version}"
    final_dir = resolve_final_model_dir(cfg.output_dir, name, cfg.overwrite_output)  # fail before hours of training
    run_dir = str(Path(cfg.checkpoint_root) / name)
    resume = resolve_resume_checkpoint(cfg.resume_from_checkpoint, run_dir)  # fail before loading the model
    fingerprint = data_fingerprint(data)
    if resume:
        check_resume_data(resume, fingerprint)
        logger.info("Resuming from checkpoint %s", resume)
    elif Path(run_dir).is_dir() and any(Path(run_dir).glob("checkpoint-*")):
        # Leftovers of an earlier (failed) run would confuse checkpoint rotation.
        stale = f"{run_dir}.old-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.warning("Moving checkpoints of a previous run aside: %s -> %s (use --resume-from-checkpoint latest to continue it)", run_dir, stale)
        os.rename(run_dir, stale)
    _write_json_atomic(Path(run_dir) / DATA_FINGERPRINT_FILE, fingerprint)  # checked by a later resume
    logger.info("Final model: %s | checkpoints: %s", final_dir, run_dir)

    precision = detect_precision(cfg.force_cpu)
    logger.info("Precision: %s", {k: v for k, v in precision.items() if k != "dtype"})
    base_model = load_base_model(cfg, precision)
    architecture = getattr(getattr(base_model, "config", None), "model_type", None)
    attn_implementation = getattr(getattr(base_model, "config", None), "_attn_implementation", None)
    target_modules, target_mode = resolve_lora_target_modules(base_model, cfg.lora_target_modules)
    model = apply_lora(base_model, cfg, target_modules=target_modules)
    lora_targets = lora_target_summary(model) or (
        [target_modules] if isinstance(target_modules, str) else list(target_modules))
    logger.info("LoRA target modules (%s, requested %r): %s", target_mode, cfg.lora_target_modules,
                ", ".join(lora_targets))
    # The base model is now local - keep every later library call off the network
    # (PEFT checks the Hub for the base config on every checkpoint save otherwise).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    train_ds = to_hf_dataset(data["train"])
    eval_ds = build_eval_dataset(data["val_tr"], data["val_en"])
    # The fast tokenizer has used its thread pool already; DataLoader workers are
    # forked afterwards and do not tokenize (silences one warning per worker).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Offline, PEFT cannot fetch the base config on every checkpoint save and
    # warns; the vocabulary is never resized here (pad = EOS), so that is fine.
    warnings.filterwarnings("ignore", message=r"Could not find a config file in .*vocabulary was not modified")
    trainer = create_trainer(model, tokenizer, train_ds, eval_ds, cfg=cfg, run_dir=run_dir, precision=precision)

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume)
    peak_gpu_memory_gb = peak_gpu_memory_reserved_gb = None
    if precision["cuda"]:
        try:
            peak_gpu_memory_gb = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            peak_gpu_memory_reserved_gb = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            logger.info("Peak GPU memory of training: %.2f GiB allocated, %.2f GiB reserved",
                        peak_gpu_memory_gb, peak_gpu_memory_reserved_gb)
        except Exception:  # pragma: no cover - driver issues
            pass

    resumed_at_step = checkpoint_global_step(resume) if resume else 0
    train_loss = whole_run_train_loss(train_result.metrics.get("train_loss"), train_result.global_step,
                                      resumed_at_step, trainer.state.log_history)

    final_metrics: Dict[str, Any] = {}
    if eval_ds is not None:
        # The final model is the best checkpoint (load_best_model_at_end): its metrics
        # were logged when it was evaluated, so no extra pass over the validation set.
        final_metrics = final_eval_metrics(trainer.state)
        if not final_metrics:
            final_metrics = trainer.evaluate()
    metrics = {
        "train_loss": train_loss,
        "best_eval_loss": trainer.state.best_metric,
        "final_eval_loss": final_metrics.get("eval_loss"),
        "final_eval_loss_turkish": final_metrics.get("eval_loss_turkish"),
        "final_eval_loss_english": final_metrics.get("eval_loss_english"),
        "final_eval_step": final_metrics.get("step"),
        "eval_loss_definition": EVAL_LOSS_DEFINITION,
    }
    duration = time.time() - started
    training_info = {
        "version": f"v{history_version}" if history_version else None,
        "model_version": f"v{name[len(MODEL_DIR_PREFIX):]}" if name.startswith(MODEL_DIR_PREFIX) else name,
        "model_dir_name": name,
        "base_model": cfg.model_name,
        "model_name": cfg.model_name,
        "model_type": "peft_lora",
        "final_loss": metrics["train_loss"],
        "metrics": metrics,
        "total_steps": train_result.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "dataset_size": summary["train_examples"],
        "train_examples": summary["train_examples"],
        "val_examples": summary["val_examples"],
        "val_examples_turkish": summary["val_examples_turkish"],
        "val_examples_english": summary["val_examples_english"],
        "dataset_files": summary["dataset"]["files"],
        "tokenization": summary["tokenization"],
        "prompt_format": summary["prompt_format"],
        "max_length": cfg.max_length,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lora_target_modules": lora_targets,
        "lora_target_modules_mode": target_mode,
        "lora_target_modules_requested": cfg.lora_target_modules,
        "architecture": architecture,
        "attn_implementation": attn_implementation,
        "trust_remote_code": cfg.trust_remote_code,
        "tokenizer": tokenizer_info(tokenizer),
        "load_in_4bit": cfg.load_in_4bit,
        "precision": "bf16" if precision["bf16"] else ("fp16" if precision["fp16"] else "fp32"),
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "validation_split": cfg.val_split_ratio,
        "seed": cfg.seed,
        "timestamp": datetime.now().isoformat(),
        "training_duration_seconds": round(duration, 1),
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "max_steps": cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else None,
        "resumed_from_checkpoint": resume,
        "resumed_at_step": resumed_at_step or None,
        "data_sha256": fingerprint["sha256"],
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
        "peak_gpu_memory_reserved_gb": peak_gpu_memory_reserved_gb,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "sort_by_length": cfg.sort_by_length,
        "log_file": Path(log_files["versioned"]).name if log_files.get("versioned") else None,
        "checkpoint_dir": run_dir,
        "model_save_path": str(final_dir),
        "library_versions": _library_versions(),
    }
    save_final_model(trainer.model, tokenizer, final_dir, training_info, overwrite=cfg.overwrite_output)

    if history_version:
        try:
            _write_json_atomic(Path(cfg.history_dir) / f"v{history_version}_info.json", training_info)
        except Exception as exc:
            logger.warning("Could not write training history info: %s", exc)
    try:
        write_metrics_csv(trainer.state.log_history, Path(run_dir) / "metrics_history.csv")
        write_metrics_csv(trainer.state.log_history, Path(cfg.log_dir) / "metrics_history.csv")
    except Exception as exc:
        logger.warning("Failed to save metrics history: %s", exc)

    logger.info("Training completed successfully in %.1f min", duration / 60)
    logger.info("Activate with: ./scripts/switch_model.sh switch %s", name)
    print(f"FINAL_MODEL_DIR={final_dir}", flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main training pipeline. Returns a process exit code."""
    global config
    _load_env()
    # Micro-batches of changing shapes fragment PyTorch's CUDA cache (reserved memory
    # up to twice the allocated peak), memory the AI worker on the same GPU then
    # cannot get. Expandable segments keep the reservation close to what is used.
    # Read when CUDA initialises (first allocation), i.e. later in this run.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args(argv)
    cfg = build_config(args)
    config = cfg
    # A redirected --log-dir also gets the error_handler log (default logs/error.log),
    # so experiments and smoke runs never write into the production logs.
    if Path(cfg.log_dir).resolve() != (PROJECT_ROOT / "logs").resolve() and not os.getenv("ERROR_LOG_FILE", "").strip():
        os.environ["ERROR_LOG_FILE"] = str(Path(cfg.log_dir) / "error.log")
    history_version = None if args.dry_run else _resolve_history_version(cfg)
    _, log_files = setup_logging(cfg, history_version, to_files=not args.dry_run)
    logger.info("Configuration: %s", {k: v for k, v in asdict(cfg).items()})

    from transformers import set_seed

    set_seed(cfg.seed)
    try:
        # A dry run never touches the GPU (get_device_name would create a CUDA context
        # next to a running worker).
        if args.dry_run:
            pass
        elif torch.cuda.is_available() and not cfg.force_cpu:
            logger.info("GPU: %s (count: %d)", torch.cuda.get_device_name(0), torch.cuda.device_count())
        else:
            logger.warning("No GPU in use - training will run on CPU (very slow)")
    except Exception as exc:  # pragma: no cover - driver issues
        logger.warning("GPU detection failed: %s", exc)

    try:
        return _run(cfg, args.dry_run, history_version, log_files)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        if "trust_remote_code" in str(exc) and not cfg.trust_remote_code:
            logger.error("The base model %s needs custom code from its repository: prefer a checkpoint that is "
                         "native in transformers (e.g. the '-hf' variant), or re-run with --trust-remote-code "
                         "(SECURITY: executes that code; inference then needs MODEL_TRUST_REMOTE_CODE=true)",
                         cfg.model_name)
        if not args.dry_run:  # a dry run writes nothing, not even the error log
            try:
                from error_handler import handle_training_error

                handle_training_error(exc, "main_training")
            except Exception:  # pragma: no cover
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())

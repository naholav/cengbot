#!/usr/bin/env python3
"""
CengBot Data Augmentation System
================================

Uses Anthropic's Claude API to generate variations of the approved
question/answer pairs in the database, to improve training diversity.

Features:
- Same row selection as database_to_training.py (active, not rejected)
- Turkish / English prompts (chosen from the row language: 'TR'/'EN'/...)
- Robust parsing of the model output (code fences, prose around the JSON,
  invalid items), link preservation check, duplicate filtering
- Retries with exponential backoff, ``retry-after`` support for rate limits,
  optional request pacing, abort after repeated failures (no runaway costs)
- Cost safety: ``--dry-run`` (no API calls), ``--max-items N``, confirmation
  prompt (``--yes`` for non-interactive runs), incremental: rows that already
  have augmented variants in the output file are not sent again
- Checkpoint (atomic) keyed by row id - safe to interrupt and resume even if
  new rows were approved in the meantime
- Output: data/cengbot_qa_augmented.jsonl (UTF-8, one JSON object per line)::

      {"id": 12, "question": ..., "answer": ..., "language": "turkish",
       "group_id": "db-12", "source": "database"}            # original
      {"id": "aug-12-1", ..., "group_id": "db-12", "source": "augmented"}

Configuration (environment, e.g. .env):
- ``ANTHROPIC_API_KEY`` or ``CLAUDE_API_KEY`` - API key (asked interactively,
  hidden, if neither is set). The key is never printed or stored.
- ``CLAUDE_MODEL`` - model id (default: claude-sonnet-5-5)
- ``CLAUDE_MAX_TOKENS`` (default 16000), ``CLAUDE_REQUEST_INTERVAL`` (min
  seconds between requests, default 0), ``CLAUDE_TEMPERATURE`` (only sent if
  set; current models reject non-default sampling parameters),
  ``CLAUDE_REFUSAL_FALLBACK`` (default on: server-side refusal fallback for
  models that support it; set to "off" to disable)

Author: naholav
"""

import argparse
import getpass
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database_models import SessionLocal, TrainingData  # noqa: E402,F401  (kept for compatibility)

# Anthropic API
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    anthropic = None
    ANTHROPIC_AVAILABLE = False

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
RETIRED_MODEL_PREFIXES = ("claude-3-", "claude-2", "claude-instant", "claude-v1")
DEFAULT_MAX_TOKENS = 16000
DEFAULT_OUTPUT_NAME = "cengbot_qa_augmented.jsonl"
CHECKPOINT_NAME = "augment_checkpoint.json"
CHECKPOINT_VERSION = 2
# Server-side refusal fallback ("fallbacks": "default") - only for models that support it.
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSAL_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5-2"}
VALID_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

_LANGUAGE_MAP = {"tr": "turkish", "turkish": "turkish", "türkçe": "turkish", "en": "english", "english": "english"}
_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")


class AugmentationAbort(RuntimeError):
    """Fatal error: continuing would only waste API credits (auth, unknown model, outages)."""


_DOTLESS_FOLD = str.maketrans({"İ": "i", "I": "i", "ı": "i"})


def _norm(text: str) -> str:
    """Case/whitespace-insensitive key; Turkish dotted/dotless i folded together."""
    return " ".join(str(text).translate(_DOTLESS_FOLD).casefold().split())


def normalize_language(value: Any, text: Optional[str] = None) -> str:
    """'TR'/'tr'/'turkish' -> 'turkish', 'EN'/... -> 'english' (guess from text otherwise)."""
    if isinstance(value, str):
        mapped = _LANGUAGE_MAP.get(value.strip().lower())
        if mapped:
            return mapped
    if text and any(ch in _TURKISH_CHARS for ch in text):
        return "turkish"
    return "english"


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return (not lowered) or lowered.startswith(("your_", "your-", "<")) or "placeholder" in lowered or lowered in ("changeme", "none", "null")


def get_api_key_from_env() -> Optional[str]:
    """API key from ANTHROPIC_API_KEY or CLAUDE_API_KEY (placeholders ignored)."""
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
        value = os.getenv(var, "")
        if value and not _is_placeholder(value):
            return value.strip()
    return None


def resolve_model_name(explicit: Optional[str] = None) -> str:
    model = (explicit or os.getenv("CLAUDE_MODEL", "") or DEFAULT_CLAUDE_MODEL).strip()
    if model.startswith(RETIRED_MODEL_PREFIXES):
        print(f"⚠️  Model '{model}' looks like a retired Claude model id; requests will likely fail. "
              f"Set CLAUDE_MODEL to a current model (default: {DEFAULT_CLAUDE_MODEL}).")
    return model


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def render_prompt(template: str, question: str, answer: str) -> str:
    """Fill {question}/{answer} without str.format (user text may contain braces)."""
    values = {"question": question, "answer": answer}
    return re.sub(r"\{(question|answer)\}", lambda m: values[m.group(1)], template)


def extract_text(response: Any) -> str:
    """Concatenate the text blocks of a Messages API response (skips thinking/other blocks)."""
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    parts: List[str] = []
    for block in content or []:
        block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
        if block_type == "text" and isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _clean_variation(item: Any) -> Optional[Dict[str, str]]:
    if not isinstance(item, dict):
        return None
    question = item.get("question", item.get("soru"))
    answer = item.get("answer", item.get("cevap"))
    if not isinstance(question, str) or not isinstance(answer, str):
        return None
    question, answer = question.strip(), answer.strip()
    if not question or not answer:
        return None
    return {"question": question, "answer": answer}


def parse_variations(text: str) -> Optional[List[Dict[str, str]]]:
    """Extract a list of {question, answer} from model output.

    Accepts a bare JSON array, an array inside ``` fences or surrounded by
    prose, or an object with a ``variations`` list. Invalid items are dropped.
    Returns None if no usable array is found.
    """
    if not text:
        return None
    cleaned = re.sub(r"```(?:json|JSON)?", "", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\[{]", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            obj = next((obj[k] for k in ("variations", "varyasyonlar", "items", "data") if isinstance(obj.get(k), list)), None)
            if obj is None:
                continue
        if isinstance(obj, list):
            items = [v for v in (_clean_variation(x) for x in obj) if v]
            if items:
                return items
    return None


def _anthropic_errors() -> SimpleNamespace:
    """Exception classes to catch (dummies when the SDK is not installed)."""

    class _Never(Exception):
        pass

    if anthropic is None:  # pragma: no cover
        return SimpleNamespace(auth=(_Never,), not_found=_Never, bad_request=_Never, rate_limit=_Never,
                               status=_Never, connection=_Never)
    return SimpleNamespace(
        auth=(anthropic.AuthenticationError, anthropic.PermissionDeniedError),
        not_found=anthropic.NotFoundError,
        bad_request=anthropic.BadRequestError,
        rate_limit=anthropic.RateLimitError,
        status=anthropic.APIStatusError,
        connection=anthropic.APIConnectionError,
    )


def _write_json_lines_atomic(path: Path, items: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_json_atomic(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class CengBotDataAugmenter:
    """Augment training data using Anthropic Claude API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_path: Optional[str] = None,
        model_name: Optional[str] = None,
        output_file: Optional[str] = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_path = Path(base_path) if base_path else PROJECT_ROOT
        self.data_dir = self.base_path / "data"
        self.output_file = Path(output_file) if output_file else self.data_dir / DEFAULT_OUTPUT_NAME
        self.checkpoint_file = self.output_file.parent / CHECKPOINT_NAME

        if client is None:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Please install with: pip install anthropic")
            api_key = api_key or get_api_key_from_env()
            if not api_key:
                raise ValueError("No Anthropic API key: set ANTHROPIC_API_KEY (or CLAUDE_API_KEY)")
            client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=600.0)
        api_key = None  # never kept on the instance
        self.client = client

        # API settings
        self.model_name = resolve_model_name(model_name)
        self.max_tokens = _env_int("CLAUDE_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        self.temperature = _env_float("CLAUDE_TEMPERATURE")  # None -> not sent
        self.top_p = None  # not sent: current models reject non-default sampling params
        self.max_retries = 3
        self.retry_delay = 2.0
        self.max_retry_delay = 60.0
        self.request_interval = _env_float("CLAUDE_REQUEST_INTERVAL", 0.0) or 0.0
        self.min_variations = 5
        self.max_variations = 20
        self.max_consecutive_failures = 5
        fallback_env = os.getenv("CLAUDE_REFUSAL_FALLBACK", "default").strip().lower()
        self.use_refusal_fallback = fallback_env not in ("", "0", "off", "false", "no", "none")
        # Optional cost lever: thinking runs adaptively by default on current
        # models and thinking tokens are billed. Paraphrasing rarely needs deep
        # reasoning, so AUGMENT_CLAUDE_EFFORT=low|medium cuts cost. Unset -> API default.
        effort = os.getenv("AUGMENT_CLAUDE_EFFORT", "").strip().lower()
        if effort and effort not in VALID_EFFORT_LEVELS:
            print(f"⚠️  Ignoring invalid AUGMENT_CLAUDE_EFFORT={effort!r} "
                  f"(valid: {', '.join(VALID_EFFORT_LEVELS)})")
            effort = ""
        self.effort = effort or None
        self._sleep = sleep
        self._last_request_at = 0.0
        self._consecutive_failures = 0
        self.stats: Counter = Counter()

        # Turkish prompt for augmentation
        self.turkish_prompt = """Sen yıllardır doğal dil işleme ve veri zenginleştirme alanında uzmanlaşmış bir AI mühendisisin. Üniversite öğrencilerinin chatbot'larla nasıl konuştuğunu çok iyi biliyorsun.

ÖNEMLİ: Bu varyasyonlar küçük bir dil modelini fine-tune etmek için kullanılacak. Varyasyonların kalitesi, çeşitliliği ve doğallığı modelin performansını doğrudan etkileyecek.

MUTLAK KURALLAR:
1. Link, isim, sayı, tarih ASLA değişmez
2. Yeni bilgi EKLEME
3. Gerçek öğrenciler gibi yaz
4. Adım adım anlatırken ASLA rakam kullanma (1,2,3 gibi)
5. Ders kodları, hoca isimleri, bölüm isimleri AYNEN korunmalı
6. Her varyasyon anlamsal olarak orijinalle %100 uyumlu olmalı
7. YAZIM HATASI KURALI: Sadece "8. Yazım hatalı" stilini kullanırken yazım hatası yap. Diğer stillerde ASLA yazım hatası yapma.
8. HİTAP KURALI: ASLA "Değerli hocam", "Sayın hocam", "Muhterem", "Kıymetli" gibi aşırı resmi hitaplar kullanma.
9. DOĞAL DİL KURALI: Öğrenciler chatbot'a yazarken doğal konuşurlar. "Merhaba", "Selam", doğrudan soru sorma veya hiç hitap kullanmama tercih edilmeli.

SORU STİLLERİ:
1. Ultra kısa
2. Günlük dil
3. Nazik ama doğal
4. Problem anlatarak
5. Çoklu soru
6. Deneyim paylaşarak
7. İngilizce terimli
8. Yazım hatalı
9. Bağlam vererek soru
10. Dolaylı/ima ederek
11. Acil/panik modda
12. Spesifik detay isteyerek
13. Kısaltmalar kullanarak
14. Belirsizlik ifadesiyle
15. Yarı resmi

CEVAP STİLLERİ:
1. Tek cümle net bilgi
2. Samimi destek
3. Detaylı açıklama
4. Öneri şeklinde
5. Adım adım
6. Karşılaştırma
7. Teknik + ekstra
8. FAQ tarzı
9. Deneyim paylaşımlı
10. Motive edici/cesaretlendirici
11. Uyarılar ve dikkat edileceklerle
12. Sebep-sonuç ilişkisiyle
13. Resmi ama anlaşılır
14. Özet + detay kombinasyonu
15. Konuşma dili ağırlıklı

ÖĞRENCİ PROFİLLERİ (Varyasyonlarda kullan):
- Hazırlık öğrencisi (kafası karışık)
- 1. sınıf (her şey yeni)
- Son sınıf (mezuniyet telaşı)
- Yüksek lisans öğrencisi (detaycı)
- Çift anadal yapan (yoğun)
- Online öğrenci (uzaktan)

Soru: {question}
Cevap: {answer}

ÇIKTI TALİMATI:
15 varyasyon üret. Her biri:
1. Farklı bir stil kombinasyonu kullansın
2. Farklı bir öğrenci profili yansıtsın
3. Çeşitlilik skoru > 0.6 olsun
4. Doğallık testi: Gerçek bir öğrenci böyle yazar mıydı?

JSON array formatında döndür. Her varyasyonda 'question' ve 'answer' alanları olsun. Sadece JSON array döndür, başka metin yazma."""

        # English prompt for augmentation
        self.english_prompt = """You are an AI engineer with years of expertise in natural language processing and data augmentation. You know very well how university students interact with chatbots.

IMPORTANT: These variations will be used to fine-tune a small language model. The quality, diversity, and naturalness of the variations will directly affect the model's performance.

ABSOLUTE RULES:
1. Links, names, numbers, dates NEVER change
2. DO NOT add new information
3. Write like real students
4. When explaining step by step, NEVER use numbers (1,2,3 etc.)
5. Course codes, professor names, department names must remain EXACTLY the same
6. Each variation must be 100% semantically aligned with the original
7. TYPO RULE: Only make typos when using "8. With typos" style. NEVER make typos in other styles.
8. GREETING RULE: NEVER use overly formal greetings like "Dear professor", "Esteemed teacher", "Respected sir/madam".
9. NATURAL LANGUAGE RULE: Students write naturally to chatbots. Use "Hi", "Hello", direct questions, or no greeting at all.

QUESTION STYLES:
1. Ultra short
2. Casual language
3. Polite but natural
4. Explaining a problem
5. Multiple questions
6. Sharing experience
7. Mixed technical terms
8. With typos
9. Providing context
10. Indirect/implying
11. Urgent/panic mode
12. Asking for specific details
13. Using abbreviations
14. With uncertainty expressions
15. Semi-formal

ANSWER STYLES:
1. Single sentence direct info
2. Friendly support
3. Detailed explanation
4. Suggestion format
5. Step by step
6. Comparison
7. Technical + extra
8. FAQ style
9. Experience sharing
10. Motivating/encouraging
11. With warnings and cautions
12. Cause-effect relationship
13. Formal but understandable
14. Summary + detail combination
15. Conversational heavy

STUDENT PROFILES (Use in variations):
- Prep student (confused)
- Freshman (everything is new)
- Senior (graduation rush)
- Graduate student (detail-oriented)
- Double major student (busy)
- Online student (remote)

Question: {question}
Answer: {answer}

OUTPUT INSTRUCTION:
Generate 15 variations. Each should:
1. Use a different style combination
2. Reflect a different student profile
3. Have diversity score > 0.6
4. Naturalness test: Would a real student write like this?

Return as JSON array. Each variation should have 'question' and 'answer' fields. Return only the JSON array, no other text."""

    # ------------------------------------------------------------------
    # Checkpoint / output
    # ------------------------------------------------------------------
    def load_checkpoint(self) -> Dict:
        """Load checkpoint data (new format: processed_ids; legacy: last_processed_index)."""
        empty = {"processed_ids": [], "output_data": None}
        if not self.checkpoint_file.exists():
            return empty
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("checkpoint is not an object")
            return data
        except (OSError, ValueError) as e:
            corrupt = self.checkpoint_file.with_name(self.checkpoint_file.name + ".corrupt")
            print(f"⚠️  Unreadable checkpoint ({e}); moved to {corrupt.name} and starting fresh")
            try:
                os.replace(self.checkpoint_file, corrupt)
            except OSError:
                pass
            return empty

    def save_checkpoint(self, processed_ids: Iterable[int], output_data: List[Dict]):
        """Save checkpoint data atomically."""
        _write_json_atomic(
            self.checkpoint_file,
            {
                "version": CHECKPOINT_VERSION,
                "processed_ids": sorted(processed_ids, key=str),
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "output_data": output_data,
            },
        )

    def save_output(self, data: List[Dict]):
        """Save output data to JSONL file (atomic replace)."""
        _write_json_lines_atomic(self.output_file, data)

    def load_existing_output(self) -> List[Dict]:
        if not self.output_file.exists():
            return []
        items: List[Dict] = []
        with open(self.output_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    items.append(obj)
        return items

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def load_training_data_from_db(self) -> List[Dict]:
        """Load approved training data from database (same filter as the training export)."""
        from database_to_training import DatabaseToTrainingConverter

        converter = DatabaseToTrainingConverter(base_path=str(self.base_path), output_file=str(self.output_file))
        records = converter.export_from_database()
        print(f"Found {len(records)} approved training records in database")
        return records

    def extract_links(self, text: str) -> List[str]:
        """Extract HTTP/HTTPS links from text (sorted, unique, trailing punctuation removed)"""
        pattern = r'https?://(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+(?:[/?#][^\s<>"{}|\\^`\[\]]*)?'
        links = {link.rstrip(".,;:!?)'\"") for link in re.findall(pattern, text or "")}
        return sorted(links)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def _fallback_active(self) -> bool:
        return self.use_refusal_fallback and self.model_name in REFUSAL_FALLBACK_MODELS

    def _throttle(self) -> None:
        if self.request_interval > 0 and self._last_request_at:
            wait = self.request_interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = time.monotonic()

    def _create_message(self, prompt: str):
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        extra_body: Dict[str, Any] = {}
        if self._fallback_active():
            kwargs["extra_headers"] = {"anthropic-beta": REFUSAL_FALLBACK_BETA}
            extra_body["fallbacks"] = "default"
        if self.effort:
            # SDK 0.57.1 predates output_config, so send it via extra_body.
            extra_body["output_config"] = {"effort": self.effort}
        if extra_body:
            kwargs["extra_body"] = extra_body
        return self.client.messages.create(**kwargs)

    def _backoff(self, attempt: int) -> float:
        return min(self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1), self.max_retry_delay)

    def _retry_after(self, error: Exception) -> Optional[float]:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) or {}
        try:
            value = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            return None
        return max(0.0, min(value, 2 * self.max_retry_delay))

    def _register_failure(self) -> None:
        self._consecutive_failures += 1
        self.stats["failed_items"] += 1
        if self._consecutive_failures >= self.max_consecutive_failures:
            raise AugmentationAbort(
                f"{self._consecutive_failures} consecutive items failed - stopping to avoid wasting API credits"
            )

    def call_api_with_retry(self, prompt: str) -> Optional[List[Dict]]:
        """Call the API with retries; returns validated variations or None."""
        errors = _anthropic_errors()
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            try:
                self._throttle()
                self.stats["api_calls"] += 1
                response = self._create_message(prompt)
            except errors.auth as e:
                raise AugmentationAbort(
                    f"Authentication failed ({type(e).__name__}) - check ANTHROPIC_API_KEY / CLAUDE_API_KEY"
                ) from None
            except errors.not_found:
                raise AugmentationAbort(
                    f"Model '{self.model_name}' not found - set CLAUDE_MODEL to a current model id"
                ) from None
            except errors.bad_request as e:
                message = str(getattr(e, "message", e))
                if self._fallback_active() and "fallback" in message.lower():
                    print("⚠️  Refusal fallback not accepted for this request - retrying without it")
                    self.use_refusal_fallback = False
                    attempt -= 1
                    continue
                print(f"⚠️  Request rejected (400): {message[:200]}")
                self._register_failure()
                return None
            except errors.rate_limit as e:
                delay = self._retry_after(e) or self._backoff(attempt)
                print(f"⏳ Rate limited (attempt {attempt}/{self.max_retries}) - waiting {delay:.1f}s")
                self._sleep(delay)
                continue
            except errors.status as e:
                status = getattr(e, "status_code", 0) or 0
                if status >= 500:
                    delay = self._backoff(attempt)
                    print(f"⚠️  API server error {status} (attempt {attempt}/{self.max_retries}) - retrying in {delay:.1f}s")
                    self._sleep(delay)
                    continue
                print(f"⚠️  API error {status}: {str(getattr(e, 'message', e))[:200]}")
                self._register_failure()
                return None
            except errors.connection as e:
                delay = self._backoff(attempt)
                print(f"⚠️  Connection error ({type(e).__name__}, attempt {attempt}/{self.max_retries}) - retrying in {delay:.1f}s")
                self._sleep(delay)
                continue

            stop_reason = getattr(response, "stop_reason", None)
            if stop_reason == "refusal":
                details = getattr(response, "stop_details", None)
                category = getattr(details, "category", None) if details is not None else None
                print(f"⚠️  Model declined this item (refusal, category={category}) - keeping the original only")
                self.stats["refusals"] += 1
                self._consecutive_failures = 0
                return None

            variations = parse_variations(extract_text(response))
            if variations and len(variations) >= self.min_variations:
                self._consecutive_failures = 0
                return variations[: self.max_variations]
            if stop_reason == "max_tokens":
                reason = "output truncated at max_tokens (raise CLAUDE_MAX_TOKENS)"
            elif not variations:
                reason = "no valid JSON array in output"
            else:
                reason = f"only {len(variations)} valid variations"
            print(f"⚠️  {reason} (attempt {attempt}/{self.max_retries})")
            self.stats["invalid_outputs"] += 1
            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt))

        self._register_failure()
        return None

    def augment_qa_pair(self, qa: Dict) -> List[Dict]:
        """Augment a single question-answer pair"""
        language = normalize_language(qa.get("language"), f"{qa.get('question', '')} {qa.get('answer', '')}")
        template = self.turkish_prompt if language == "turkish" else self.english_prompt
        prompt = render_prompt(template, qa["question"], qa["answer"])

        variations = self.call_api_with_retry(prompt)
        if not variations:
            return []

        original_links = self.extract_links(qa["answer"])
        seen = {(_norm(qa["question"]), _norm(qa["answer"]))}
        formatted: List[Dict] = []
        for var in variations:
            if self.extract_links(var["answer"]) != original_links:
                self.stats["link_mismatch"] += 1
                continue
            key = (_norm(var["question"]), _norm(var["answer"]))
            if key in seen:
                self.stats["duplicate_variations"] += 1
                continue
            seen.add(key)
            formatted.append({"question": var["question"], "answer": var["answer"], "language": language})
        return formatted

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def process_all_data(self, max_items: Optional[int] = None, dry_run: bool = False, reset: bool = False) -> Dict:
        """Augment every approved row that has no variations yet.

        ``dry_run`` only reports what would be done (no API calls, no writes).
        """
        from database_to_training import db_group_id

        input_data = self.load_training_data_from_db()
        checkpoint = {"processed_ids": [], "output_data": None} if reset else self.load_checkpoint()

        if checkpoint.get("output_data") is not None:
            output_data = list(checkpoint["output_data"])
            processed = set(checkpoint.get("processed_ids") or [])
            if not processed and "last_processed_index" in checkpoint:  # legacy checkpoint
                last = int(checkpoint.get("last_processed_index", -1))
                processed = {qa["id"] for qa in input_data[: last + 1]}
                print("⚠️  Legacy checkpoint found - resuming by position (best effort)")
        else:
            output_data = [] if reset else self.load_existing_output()
            processed = {
                int(e["group_id"][3:])
                for e in output_data
                if e.get("source") == "augmented" and str(e.get("group_id", "")).startswith("db-") and str(e["group_id"][3:]).isdigit()
            }

        pending = [qa for qa in input_data if qa["id"] not in processed]
        if max_items is not None:
            pending = pending[: max(0, int(max_items))]
        plan = {
            "model": self.model_name,
            "approved_records": len(input_data),
            "already_augmented": len(input_data) - len([qa for qa in input_data if qa["id"] not in processed]),
            "to_process": len(pending),
            "max_api_calls": len(pending) * self.max_retries,
            "output_file": str(self.output_file),
        }
        if dry_run:
            print("📋 Augmentation plan (dry run, no API calls):")
            for key, value in plan.items():
                print(f"   {key}: {value}")
            return plan

        print(f"🚀 Starting CengBot Data Augmentation ({len(pending)} records, model {self.model_name})")
        done = 0
        try:
            for n, qa in enumerate(pending, 1):
                group = db_group_id(qa["id"])
                print(f"\n🔄 Processing: {n}/{len(pending)} - {qa['language']} question (id {qa['id']})")
                variations = self.augment_qa_pair(qa)
                group_items = [{
                    "id": qa["id"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "language": normalize_language(qa.get("language"), qa["question"]),
                    "group_id": group,
                    "source": "database",
                }]
                for k, var in enumerate(variations, 1):
                    var.update({"id": f"aug-{qa['id']}-{k}", "group_id": group, "source": "augmented"})
                    group_items.append(var)
                # Drop a previous copy of this original (e.g. from an earlier failed attempt).
                key = (_norm(qa["question"]), _norm(qa["answer"]))
                output_data = [
                    e for e in output_data
                    if not (e.get("source") == "database" and (_norm(e.get("question", "")), _norm(e.get("answer", ""))) == key)
                ]
                output_data.extend(group_items)
                processed.add(qa["id"])
                done += 1
                self.stats["processed_items"] = done
                if variations:
                    self.stats["augmented_items"] += 1
                    print(f"✨ Generated {len(variations)} variations")
                    if len(variations) < 15:
                        print(f"⚠️  Only {len(variations)} valid variations (expected 15)")
                else:
                    print("❌ No variations generated - kept the original only")
                if n % 10 == 0:
                    self.save_checkpoint(processed, output_data)
                if n % 50 == 0:
                    self.save_output(output_data)
                    print(f"💾 Auto-save: {len(output_data)} records saved to file")
        finally:
            # Persist progress (also on Ctrl+C / abort) so a resume never re-pays for done items.
            # Nothing done (e.g. invalid API key on the first item): leave the files untouched.
            if done:
                self.save_checkpoint(processed, output_data)
                self.save_output(output_data)

        complete = all(qa["id"] in processed for qa in input_data)
        if complete and self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        summary = {**plan, "processed": done, "total_output_records": len(output_data), "complete": complete,
                   "stats": dict(self.stats)}
        print("\n🎉 Data augmentation finished")
        print(f"   Processed: {done} | Output records: {len(output_data)} | File: {self.output_file}")
        print(f"   Stats: {dict(self.stats)}")
        return summary


class _NoClient:
    """Placeholder client for dry runs - any API call is a bug."""

    class _Messages:
        def create(self, **kwargs):  # pragma: no cover
            raise RuntimeError("API call attempted in dry-run mode")

    messages = _Messages()


def _load_env() -> None:
    """Load the project .env (never overrides variables that are already set)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def main(argv: Optional[List[str]] = None) -> int:
    """Command line entry point (interactive by default, see --help)."""
    parser = argparse.ArgumentParser(description="Augment approved training data with the Claude API")
    parser.add_argument("--model", default=None, help=f"Claude model id (default: env CLAUDE_MODEL or {DEFAULT_CLAUDE_MODEL})")
    parser.add_argument("--max-items", type=int, default=None, help="Process at most N database rows (cost cap)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done; no API calls, no writes")
    parser.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation")
    parser.add_argument("--output", default=None, help=f"Output JSONL (default: data/{DEFAULT_OUTPUT_NAME})")
    parser.add_argument("--base-path", default=None, help="Project root (default: auto-detected)")
    parser.add_argument("--reset", action="store_true", help="Ignore checkpoint/existing output and re-augment everything")
    args = parser.parse_args(argv)
    _load_env()

    print("🤖 CengBot Data Augmentation System")
    print("=" * 50)

    if args.dry_run:
        augmenter = CengBotDataAugmenter(base_path=args.base_path, model_name=args.model, output_file=args.output,
                                         client=_NoClient())
        try:
            augmenter.process_all_data(max_items=args.max_items, dry_run=True, reset=args.reset)
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}")
            return 1
        return 0

    if not ANTHROPIC_AVAILABLE:
        print("❌ ERROR: Anthropic package not installed (pip install anthropic)")
        return 1

    print("\n⚠️  IMPORTANT: this uses Anthropic's Claude API, which incurs costs.")
    print("- The API key is read from ANTHROPIC_API_KEY / CLAUDE_API_KEY or asked (hidden); it is never stored or printed")
    print("- Progress is checkpointed - the process can be interrupted safely")

    api_key = get_api_key_from_env()
    if not api_key:
        if not sys.stdin.isatty():
            print("❌ ERROR: no API key in ANTHROPIC_API_KEY / CLAUDE_API_KEY and no terminal to ask for one")
            return 1
        api_key = getpass.getpass("\n🔑 Enter your Anthropic API key (input hidden): ").strip()
        if not api_key:
            print("❌ ERROR: API key is required")
            return 1
    if not api_key.startswith("sk-ant-"):
        print("⚠️  The API key does not look like an Anthropic key (expected prefix 'sk-ant-')")

    try:
        augmenter = CengBotDataAugmenter(api_key=api_key, base_path=args.base_path, model_name=args.model,
                                         output_file=args.output)
    finally:
        api_key = None

    plan = augmenter.process_all_data(max_items=args.max_items, dry_run=True, reset=args.reset)
    if plan["to_process"] == 0:
        print("✅ Nothing to augment")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("❌ Refusing to spend API credits without confirmation - pass --yes")
            return 1
        answer = input(f"\n❓ Send {plan['to_process']} records to {augmenter.model_name}? (y/N): ").strip().lower()
        if answer != "y":
            print("❌ Process cancelled")
            return 0

    try:
        augmenter.process_all_data(max_items=args.max_items, reset=args.reset)
    except KeyboardInterrupt:
        saved = "progress saved" if augmenter.stats.get("processed_items") else "nothing processed yet"
        print(f"\n⚠️  Interrupted by user - {saved}, run again to resume")
        return 130
    except AugmentationAbort as e:
        saved = "progress saved" if augmenter.stats.get("processed_items") else "nothing processed, no files written"
        print(f"\n❌ Aborted: {e} ({saved})")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        return 1
    print("\n✅ Data augmentation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

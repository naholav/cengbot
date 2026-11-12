"""
Prompt guard for CengBot (Turkish + English)
=============================================

A small, offline, dependency-free filter that sits in front of the language
model (the single choke point is ``ai_model_worker``; the standalone bot can
call it too). It is deliberately rule based: no ML models, no network, microseconds per
message.

Input side (``guard_input``):
  * Unicode NFKC normalisation, removal of zero-width / control / bidi
    characters, whitespace collapsing.
  * Length cap (``PROMPT_GUARD_MAX_CHARS``, default 1000) -> polite refusal.
  * Neutralisation of things that can forge turns in the plain-text prompt
    template (``"{system}\\n\\nStudent: {message}\\nAssistant:"`` and its Turkish
    variant ``Öğrenci:/Asistan:``): role markers such as ``Student:``,
    ``Asistan:``, ``System:`` lose their colon, the chat/special tokens of
    the common open-weight model families are removed (Llama ``<|eot_id|>``,
    ChatML ``<|im_start|>``, Llama-2/Mistral ``[INST]`` / ``<<SYS>>`` /
    ``<s>`` / ``</s>``, Gemma ``<start_of_turn>`` / ``<eos>``, Phi/GLM
    ``<|user|>`` / ``[gMASK]``, DeepSeek full-width ``<｜User｜>``, reasoning
    ``<think>`` tags), ``###`` is reduced.
  * A small weighted rule set that detects common jailbreak / prompt-injection
    phrasings in both languages, including ASCII-folded Turkish ("talimatlari
    unut"), leetspeak ("1gn0re") and spaced-out letters ("j a i l b r e a k").
    A message is blocked when its score reaches ``BLOCK_THRESHOLD``; the model
    is then not called at all and the user gets a refusal in their language.

Output side (``guard_output``):
  * Drop reasoning blocks (``<think>...</think>``, also an unterminated
    ``<think>`` or a bare ``</think>`` whose opening tag was in the prompt) of
    reasoning models (DeepSeek-R1 distills, Qwen3) before anything else.
  * Cut the answer at the first generated end-of-turn / new-turn special token
    (``<|eot_id|>``, ``<|im_end|>``, ``<end_of_turn>``, ``</s>``, ``<|user|>`` ...)
    and at the first generated turn marker, remove leaked special tokens.
  * Replace the answer with a safe fallback if it contains a substantial
    verbatim chunk of a registered protected text (the system prompts; the
    model handler registers them via ``register_protected_text``).

False positives are worse than misses here (students ask real questions), so
single weak signals never block on their own. Design decisions worth knowing:
  * Benign role play is allowed: "Act as a TA and explain recursion" passes
    (only "act as" + unrestricted / no-rules wording is blocked).
  * "ignore case in SQL", "git ignore previous commits", "CPU skips
    instructions" pass: the ignore-rules need a prompt-specific object
    (instructions / prompt / guidelines ...) *and* a qualifier such as
    previous / above / your / all.
  * Turkish "unuttum" (I forgot) never matches the imperative "unut".

Enable/disable with ``PROMPT_GUARD_ENABLED`` (default: enabled). Blocked
attempts are logged at WARNING with a short preview only.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "GuardResult",
    "guard_input",
    "guard_output",
    "is_enabled",
    "normalize_text",
    "neutralize_markers",
    "strip_special_tokens",
    "strip_reasoning",
    "cut_at_end_of_turn",
    "cut_at_turn_markers",
    "find_turn_marker",
    "contains_protected_text",
    "register_protected_text",
    "fold_text",
    "detect_language",
    "score_injection",
    "BLOCK_THRESHOLD",
    "DEFAULT_MAX_CHARS",
]

DEFAULT_MAX_CHARS = 1000
BLOCK_THRESHOLD = 3
_PREVIEW_CHARS = 60

_FALSE_VALUES = {"0", "false", "no", "off", "disabled", "disable"}


@dataclass
class GuardResult:
    allowed: bool
    text: str
    reason: Optional[str] = None
    refusal: Optional[str] = None


# ---------------------------------------------------------------------------
# Configuration helpers (read at call time so tests / ops can toggle them)
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    value = os.getenv("PROMPT_GUARD_ENABLED")
    if value is None or not value.strip():
        return True
    return value.strip().lower() not in _FALSE_VALUES


def _max_chars() -> int:
    value = os.getenv("PROMPT_GUARD_MAX_CHARS")
    if value:
        try:
            parsed = int(value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
        logger.warning("Invalid PROMPT_GUARD_MAX_CHARS=%r, using %d", value, DEFAULT_MAX_CHARS)
    return DEFAULT_MAX_CHARS


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

_REFUSAL = {
    "tr": (
        "Üzgünüm, bu isteğe yardımcı olamam. Çukurova Üniversitesi Bilgisayar "
        "Mühendisliği bölümüyle ilgili sorularını memnuniyetle yanıtlarım."
    ),
    "en": (
        "Sorry, I can't help with that request. I'm happy to answer your "
        "questions about Çukurova University Computer Engineering."
    ),
}
_TOO_LONG = {
    "tr": "Mesajın çok uzun ({length} karakter). Lütfen sorunu en fazla {limit} karakterle yeniden yaz.",
    "en": "Your message is too long ({length} characters). Please shorten your question to at most {limit} characters.",
}
_EMPTY = {
    "tr": "Lütfen sorunu yazı olarak gönder.",
    "en": "Please type your question as text.",
}
_OUTPUT_FALLBACK = {
    "tr": "Üzgünüm, bu konuda yardımcı olamıyorum. Bölümle ilgili başka bir sorunu yanıtlamaktan memnuniyet duyarım.",
    "en": "Sorry, I can't help with that. I'm happy to answer other questions about the department.",
}


def _lang(language: Optional[str], text: str) -> str:
    if language:
        value = str(language).strip().lower()
        if value in ("tr", "turkish", "türkçe", "turkce"):
            return "tr"
        if value in ("en", "english", "ingilizce"):
            return "en"
    return detect_language(text)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_HSPACE_RE = re.compile(r"[^\S\n]+")
_NEWLINES_RE = re.compile(r" *\n\s*")


def normalize_text(text: str) -> str:
    """NFKC, drop invisible/control characters, collapse whitespace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    out = []
    for ch in text:
        if ch == "\n":
            out.append(ch)
            continue
        if ch in "\r\t\v\f":
            out.append("\n" if ch == "\r" else " ")
            continue
        cat = unicodedata.category(ch)
        if cat in ("Zl", "Zp"):
            out.append("\n")
        elif cat in ("Cc", "Cf", "Cs", "Co", "Cn"):
            # control, format (zero-width, bidi overrides, soft hyphen, BOM),
            # surrogates, private use, unassigned
            continue
        else:
            out.append(ch)
    text = "".join(out).replace("\r\n", "\n")
    text = _HSPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n", text)
    return text.strip()


_TR_FOLD = str.maketrans({
    "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g", "ı": "i", "I": "i", "İ": "i",
    "ö": "o", "Ö": "o", "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
    "â": "a", "Â": "a", "î": "i", "Î": "i", "û": "u", "Û": "u",
})


def fold_text(text: str) -> str:
    """Lower-case + ASCII-fold Turkish letters (for matching only)."""
    text = (text or "").translate(_TR_FOLD).lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


_LEET_CHARS = "0134578@$€"
_LEET_I = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                         "8": "b", "@": "a", "$": "s", "€": "e"})
_LEET_L = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t",
                         "8": "b", "@": "a", "$": "s", "€": "e"})
_TOKEN_RE = re.compile(r"\S+")
_BETWEEN_LETTERS_RE = re.compile(r"(?<=[a-z])[!|](?=[a-z])")


def _deleet(folded: str, table=_LEET_I, between: str = "i") -> str:
    def fix(match: "re.Match[str]") -> str:
        token = match.group(0)
        if any(c.isalpha() for c in token) and any(c in _LEET_CHARS for c in token):
            token = token.translate(table)
        return token
    folded = _TOKEN_RE.sub(fix, folded)
    return _BETWEEN_LETTERS_RE.sub(between, folded)


_SPACED_LETTERS_RE = re.compile(r"\b(?:[a-z0-9][ ._\-*]){3,}[a-z0-9]\b")


def _views_for(folded: str) -> Tuple[str, str]:
    words = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    despaced = _SPACED_LETTERS_RE.sub(lambda m: re.sub(r"[ ._\-*]", "", m.group(0)), folded)
    despaced = re.sub(r"[^a-z0-9]+", " ", despaced).strip()
    return words, despaced


def _matching_views(text: str) -> Tuple[List[str], List[str]]:
    """Return (word views, compact views) used for rule matching.

    Word views: ASCII-folded, de-leeted ("1" read as "i" and as "l"), punctuation
    removed, plus a variant where spaced-out letters are joined. Compact views have
    all separators removed.
    """
    folded = fold_text(text)
    views: List[str] = []
    for table, between in ((_LEET_I, "i"), (_LEET_L, "l")):
        for view in _views_for(_deleet(folded, table, between)):
            if view not in views:
                views.append(view)
    compact = []
    for view in views:
        squeezed = view.replace(" ", "")
        if squeezed not in compact:
            compact.append(squeezed)
    return views, compact


# ---------------------------------------------------------------------------
# Special tokens and role markers
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(
    # <|begin_of_text|> <|eot_id|> (Llama 3), <|im_start|> <|im_end|> <|endoftext|> (ChatML: Qwen, SmolLM,
    # Phi-4), <|user|> <|end|> (Phi-3, GLM, OLMo, Falcon3), <|start_of_role|> (Granite) and DeepSeek's
    # full-width-bar tokens <｜User｜> <｜end▁of▁sentence｜> (NFKC maps "｜" to "|", raw text may still have it)
    r"<[|｜][^<>|｜\n]{1,64}[|｜]>"
    r"|\[/?(?:INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_RESULTS|TOOL_CALLS)\]"  # Llama 2 / Mistral
    r"|<</?SYS>>"                     # <<SYS>> <</SYS>> (Llama 2)
    r"|</?s>"                         # <s> </s> (SentencePiece BOS/EOS: Llama 2, Mistral)
    r"|</?(?:start|end)_of_turn>"     # <start_of_turn> <end_of_turn> (Gemma)
    r"|<(?:bos|eos)>"                 # <bos> <eos> (Gemma)
    r"|</?(?:think|tool_call|tool_response)>"  # reasoning / tool tags (DeepSeek-R1, Qwen3, Hermes)
    r"|\[[gs]MASK\]|<[se]op>",         # [gMASK] <sop> <eop> (GLM)
    re.IGNORECASE,
)
_HASHES_RE = re.compile(r"#{3,}")

_ROLE_WORDS = (
    r"st[uü]dent|[öo][ğg]renc[iıİ]|ass[iı]stant|as[iı]stan|s[yi]stem|s[iı]stem"
    r"|user|kullan[iı]c[iı]|human"
)
_ROLE_MARKER_RE = re.compile(r"(?<![\w])(" + _ROLE_WORDS + r")(\s*):", re.IGNORECASE)
# Output side: the model continues the dialogue with a new turn.
_TURN_MARKER_RE = re.compile(r"(?<![\w])(?:" + _ROLE_WORDS + r")\s*:", re.IGNORECASE)
_LEADING_ANSWER_MARKER_RE = re.compile(r"^\s*(?:ass[iı]stant|as[iı]stan)\s*:\s*", re.IGNORECASE)


def strip_special_tokens(text: str) -> str:
    """Remove chat/special tokens that could open or close turns."""
    text = _SPECIAL_TOKEN_RE.sub(" ", text or "")
    return _HASHES_RE.sub("#", text)


_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove the reasoning of reasoning models (DeepSeek-R1 distills, Qwen3 ...).

    * ``<think>...</think>`` blocks are removed;
    * a ``</think>`` without an opening tag before it means the chat template
      opened the block in the prompt: everything up to it is reasoning;
    * an unterminated ``<think>`` (generation stopped while thinking): everything
      from it on is reasoning.
    """
    text = text or ""
    if "think>" not in text.lower():
        return text
    first_close = _THINK_CLOSE_RE.search(text)
    first_open = _THINK_OPEN_RE.search(text)
    if first_close and (first_open is None or first_close.start() < first_open.start()):
        text = text[first_close.end():]
    text = _THINK_BLOCK_RE.sub(" ", text)
    unterminated = _THINK_OPEN_RE.search(text)
    if unterminated:
        text = text[:unterminated.start()]
    return _THINK_CLOSE_RE.sub(" ", text)


# Output side: a generated end-of-turn / new-turn token ends the answer (whatever follows is
# a new dialogue turn). Granite's <|end_of_role|> follows the role name at the START of a
# turn and is only stripped.
_END_OF_TURN_RE = re.compile(
    r"<[|｜](?:eot_id|end_of_text|endoftext|im_end|im_start|end|user|assistant|system|observation"
    r"|start_header_id|start_of_role|end▁of▁sentence)[|｜]>"
    r"|<(?:start|end)_of_turn>|</s>|<eos>|\[INST\]",
    re.IGNORECASE,
)


def cut_at_end_of_turn(text: str) -> str:
    """Drop everything from the first end-of-turn / new-turn special token that follows
    some answer text (leading special tokens are skipped)."""
    text = text or ""
    for match in _END_OF_TURN_RE.finditer(text):
        if strip_special_tokens(text[:match.start()]).strip():
            return text[:match.start()]
    return text


def neutralize_markers(text: str) -> str:
    """Remove special tokens and defuse ``Role:`` markers (``Student:`` -> ``Student -``)."""
    text = strip_special_tokens(text)
    text = _ROLE_MARKER_RE.sub(lambda m: m.group(1) + " -", text)
    return _HSPACE_RE.sub(" ", text).strip()


def find_turn_marker(text: str) -> Optional[int]:
    """Index of the first role/turn marker (``Student:``, ``Öğrenci:`` ...) or None."""
    match = _TURN_MARKER_RE.search(text or "")
    return match.start() if match else None


def cut_at_turn_markers(text: str) -> str:
    """Drop everything from the first generated turn marker on."""
    text = _LEADING_ANSWER_MARKER_RE.sub("", text or "")
    index = find_turn_marker(text)
    if index is not None:
        text = text[:index]
    return text.strip()


# ---------------------------------------------------------------------------
# Language guess (no external deps)
# ---------------------------------------------------------------------------

_TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TR_WORDS = {
    "ve", "bir", "bu", "su", "ne", "nedir", "neden", "nasil", "hangi", "kim", "kimdir",
    "nerede", "nereden", "nereye", "zaman", "mi", "mu", "misin", "miyim", "midir", "var",
    "yok", "icin", "ile", "ama", "veya", "da", "de", "ki", "ben", "sen", "bana", "sana",
    "merhaba", "selam", "tesekkur", "tesekkurler", "sagol", "lutfen", "evet", "hayir",
    "gibi", "daha", "cok", "olan", "olarak", "ders", "dersi", "dersler", "hoca", "hocasi",
    "bolum", "bolumu", "sinav", "sinavi", "ogrenci", "universite", "nelerdir", "kac",
    "kadar", "sonra", "once", "simdi", "artik", "degil", "degilsin", "unut", "goster",
    "talimat", "talimatlari", "kurallari", "yaz", "soyle", "yap", "olur", "alabilir",
    "nasilsin", "gunaydin", "aksamlar", "gunler",
}
_EN_WORDS = {
    "the", "is", "are", "what", "how", "who", "where", "when", "which", "why", "can",
    "could", "do", "does", "did", "i", "you", "my", "your", "a", "an", "of", "to", "in",
    "for", "on", "and", "or", "with", "about", "please", "hello", "hi", "thanks", "thank",
    "there", "this", "that", "it", "be", "will", "would", "should", "have", "has", "get",
    "me", "all", "previous", "instructions", "ignore", "tell", "show", "now",
}
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def detect_language(text: str) -> str:
    """Cheap TR/EN guess used for refusal messages ('tr' or 'en')."""
    tr_score = 0.0
    en_score = 0.0
    for word in _WORD_RE.findall(text or ""):
        base = re.split(r"['’]", word)[0]
        folded = fold_text(base)
        if folded == "cukurova":
            continue
        if any(c in _TR_CHARS for c in base):
            # Capitalised words with Turkish letters are often names (Dr. Şahin)
            tr_score += 2.0 if (base[0].islower() or base.isupper()) else 0.5
        if folded in _TR_WORDS:
            tr_score += 1.0
        if folded in _EN_WORDS:
            en_score += 1.0
    if tr_score == en_score == 0:
        return "en"
    return "tr" if tr_score >= en_score else "en"


# ---------------------------------------------------------------------------
# Injection rules
# ---------------------------------------------------------------------------

# English building blocks (on the ASCII-folded "words" view)
_EN_IGNORE_VERB = (r"(?:ignore|disregard|forget|skip|override|bypass|overlook|neglect|drop|abandon"
                   r"|discard|erase|delete|clear|remove|throw away|set aside)")
_EN_FILL = r"(?:(?:the|of|these|those|this|that|and|my)\s+)*"
_EN_KEY = (r"(?:previous|prior|above|earlier|preceding|foregoing|initial|original|old|your"
           r"|system|all|any|every|existing|hidden)")
_EN_STRONG_NOUN = (r"(?:instructions?|prompts?|directives?|guidelines?|programming|guardrails?"
                   r"|system\s+(?:prompt|message)s?|training)")
_EN_WEAK_NOUN = r"(?:rules?|commands?|restrictions?|constraints?|filters?|limitations?|polic(?:y|ies))"
_EN_REVEAL_VERB = (r"(?:reveal|show|print|display|output|repeat|tell|give|leak|share|expose|dump|write"
                   r"|list|recite|spell\s+out|paste|copy|send|what\s+(?:is|are|was|were)|whats)")
_EN_MODS = (r"(?:(?:me|us|the|your|its|all|full|entire|complete|exact|original|initial|hidden|secret"
            r"|internal|underlying|first|above|previous|verbatim|real|actual|current)\s+)")
_EN_PROMPT_NOUN = (r"(?:system\s+prompt|system\s+message|system\s+instructions|initial\s+prompt"
                   r"|hidden\s+prompt|original\s+prompt|secret\s+prompt|pre\s?prompt|meta\s+prompt"
                   r"|hidden\s+instructions)")
_EN_YOUR_NOUN = (r"(?:instructions|prompt|rules|guidelines|directives|configuration|programming"
                 r"|training\s+data|system\s+prompt)")
_EN_BAD_PERSONA = (r"(?:unrestricted|unfiltered|uncensored|jailbroken|evil|unethical|amoral|unlimited"
                   r"|lawless|rogue|unbound|unchained|limitless|without\s+(?:any\s+)?(?:restrictions|rules"
                   r"|filters|limits|guidelines|censorship|ethics)|free\s+from|no\s+(?:rules|limits|restrictions|filters))")

# Turkish building blocks (ASCII-folded)
# "Disregard" verbs. "uyma" is also the noun "compliance" ("kurallara uyma
# zorunluluğu var mı?"), so it does not count when a noun follows it.
_TR_IGNORE_VERB = (r"(?:unut(?:un|man|mani|malisin|acaksin)?|yok\s+say(?:in|man|acaksin)?"
                   r"|gormezden\s+gel(?:in)?|goz\s+ardi\s+et(?:in)?|dikkate\s+alma(?:yin|yacaksin)?"
                   r"|umursama(?:yin)?|takma|bosver|bos\s+ver|aldirma|cigne|ihlal\s+et|es\s+gec"
                   r"|uyma(?:yin|yacaksin)?\b(?!\s+(?:zorunlu|yukumlu|sart|gerek|mecbur|kosul|durum|konu"
                   r"|sorumlu|oran|hakk|bilinc|kural)\w*))\b")
# "Cancel/delete" verbs: an override of instructions/rules, but ordinary actions on a
# (shell/git) command - "önceki commit komutunu iptal et" is a programming question.
_TR_EDIT_VERB = r"(?:iptal\s+et|gecersiz\s+(?:say|kil)|sil|devre\s+disi\s+birak)\b"
_TR_VERB = rf"(?:{_TR_IGNORE_VERB}|{_TR_EDIT_VERB})"
_TR_STRONG_NOUN = r"(?:talimat|yonerge|direktif|prompt|istem|sistem\s+mesaj)\w*"
_TR_QUAL = (r"(?:onceki|yukaridaki|yukardaki|tum|butun|eski|ilk|verilen|sana\s+verilen|senin|kendi"
            r"|gizli|sistem|mevcut|baslangic)")
_TR_REVEAL_VERB = (r"(?:goster|yaz|soyle|paylas|tekrarla|listele|yazdir|ifsa|acikla|dok|ver|nedir|ne\b"
                   r"|neler|nelerdir)\w*")
# Qualifiers that make "talimat/yönerge" the bot's own instructions ...
_TR_OWN_QUAL = r"(?:sistem|gizli|senin|sana\s+verilen|kendi)"
# ... and ones that are ordinary with documents ("ilk talimatlar", "staj iç yönergesi"):
# they only count with a prompt noun or a second-person marker.
_TR_WEAK_QUAL = r"(?:ilk|baslangic|orijinal|asil|ic)"
_TR_PROMPT_NOUN = r"(?:prompt|istem)\w*"
_TR_INSTR_NOUN = r"(?:prompt|istem|talimat|yonerge|direktif)\w*"
_TR_SECOND_PERSON = r"(?:senin|sana|kendi)"
_TR_BAD_ADJ = (r"(?:kisitlamasiz|sansursuz|filtresiz|kuralsiz|sinirsiz|limitsiz|etik\s+disi|ahlaksiz"
               r"|kotu\s+niyetli|serbest|ozgur)")


def _r(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern)


# (name, weight, compiled regex) matched against the words view and the
# despaced view. Each rule counts once.
_RULES: List[Tuple[str, int, "re.Pattern[str]"]] = [
    # --- English: override previous instructions
    ("en_ignore_instructions", 4, _r(rf"\b{_EN_IGNORE_VERB}\s+{_EN_FILL}{_EN_KEY}\s+(?:\w+\s+){{0,3}}{_EN_STRONG_NOUN}\b")),
    ("en_ignore_rules", 2, _r(rf"\b{_EN_IGNORE_VERB}\s+{_EN_FILL}{_EN_KEY}\s+(?:\w+\s+){{0,2}}{_EN_WEAK_NOUN}\b")),
    ("en_ignore_your", 4, _r(r"\b(?:ignore|disregard|forget|break|bypass|override|violate|abandon|drop|escape"
                             r"|circumvent|throw\s+away)\s+(?:all\s+|of\s+|any\s+)*your\s+(?:\w+\s+){0,2}"
                             r"(?:rules|restrictions|filters|limitations|constraints|policies|guidelines|training"
                             r"|programming|instructions|safeguards|guardrails|ethics|principles|prompt)\b")),
    ("en_ignore_above", 3, _r(r"\b(?:ignore|disregard|forget)\s+(?:all\s+of\s+|all\s+)?(?:the\s+|what\s+is\s+)?"
                              r"(?:above|everything\s+(?:above|before|prior)|all\s+of\s+the\s+above|prior\s+text"
                              r"|text\s+above)\b")),
    ("en_forget_told", 4, _r(r"\b(?:ignore|disregard|forget)\s+(?:all\s+|everything\s+|anything\s+|what\s+)?"
                             r"(?:that\s+)?you\s+(?:were|have\s+been|ve\s+been|had\s+been)\s+"
                             r"(?:told|given|taught|instructed|programmed)\b")),
    ("en_forget_everything", 2, _r(r"\bforget\s+(?:about\s+)?(?:everything|all)\b")),
    # --- English: prompt extraction
    ("en_reveal_system_prompt", 4, _r(rf"\b{_EN_REVEAL_VERB}\s+{_EN_MODS}{{0,4}}{_EN_PROMPT_NOUN}\b")),
    ("en_reveal_your_prompt", 4, _r(rf"\b{_EN_REVEAL_VERB}\s+{_EN_MODS}{{0,3}}(?:your\s+{_EN_MODS}{{0,2}}{_EN_YOUR_NOUN}\b"
                                    rf"|{_EN_YOUR_NOUN}\s+(?:that\s+)?you\s+(?:were|have\s+been"
                                    rf"|ve\s+been|had\s+been)\s+(?:given|told|programmed|trained|configured)\b)")),
    ("en_repeat_above", 3, _r(r"\b(?:repeat|print|output|show|copy|echo)\s+(?:(?:all|the|everything|every|word|words"
                              r"|text|lines?|of)\s+){0,3}(?:above|before\s+this|preceding|verbatim)\b")),
    ("en_system_prompt_mention", 2, _r(r"\b(?:system\s+prompt|initial\s+prompt|hidden\s+prompt|pre\s?prompt)\b")),
    # --- English: persona / mode switching
    ("en_bad_persona", 4, _r(rf"\b(?:you\s+(?:are|re|r)|act\s+as|acting\s+as|become|be|pretend\s+(?:to\s+be|you\s+are)"
                             rf"|roleplay\s+as|role\s+play\s+as|play\s+the\s+role\s+of|behave\s+(?:like|as)|simulate"
                             rf"|respond\s+as|answer\s+as)\s+(?:an?\s+|the\s+)?(?:\w+\s+){{0,2}}{_EN_BAD_PERSONA}")),
    ("en_you_are_now", 2, _r(r"\byou\s+(?:are|re|r)\s+(?:now|no\s+longer)\b|\bfrom\s+now\s+on\b.{0,40}\byou\b")),
    ("en_not_bound", 3, _r(r"\b(?:not|no\s+longer|never)\s+(?:be\s+)?(?:bound|restricted|limited|constrained)\s+by\b")),
    ("en_roleplay", 1, _r(r"\b(?:act\s+as|pretend|roleplay|role\s+play|play\s+the\s+role|impersonate)\b")),
    ("en_jailbreak", 4, _r(r"\bjail\s?break(?:s|ed|ing)?\b")),
    ("en_bad_mode", 4, _r(r"\b(?:unrestricted|unfiltered|uncensored|jailbroken|jailbreak|evil|dan|chaos|opposite"
                          r"|god)\s+mode\b")),
    ("en_developer_mode", 1, _r(r"\b(?:developer|dev)\s+mode\b")),
    ("en_developer_mode_ai", 2, _r(r"\b(?:chatgpt|gpt|claude|llama|qwen|mistral|gemma|deepseek|assistant|bot|model|ai)\b.{0,30}\bdeveloper\s+mode\b"
                                   r"|\bdeveloper\s+mode\s+(?:enabled|activated)\b")),
    ("en_do_anything_now", 4, _r(r"\bdo\s+anything\s+now\b")),
    ("en_dan", 3, _r(r"\b(?:you\s+(?:are|re)|act\s+as|be|become|pretend\s+to\s+be|called|named|as)\s+dan\b"
                     r"|\bdan\s+(?:mode|prompt|jailbreak|modu)\b")),
    ("en_uncensored", 2, _r(r"\b(?:uncensored|unfiltered|unrestricted|jailbroken)\b")),
    ("en_no_restrictions", 1, _r(r"\b(?:without|no|zero|free\s+of|free\s+from)\s+(?:any\s+)?(?:restrictions|filters"
                                 r"|filtering|censorship|limitations|guardrails|rules|guidelines|ethics|morals)\b")),
    # "remove (all) filters" is an everyday action (Excel, mail, images): with "remove"
    # only content/safety filters count.
    ("en_disable_safety", 2, _r(r"\b(?:(?:bypass|disable|turn\s+off|deactivate|circumvent|evade|get\s+around|break)"
                                r"\s+(?:(?:the|all|any|its|my|these|content|safety)\s+){0,3}(?:filters?|filtering"
                                r"|guardrails|safeguards|censorship|alignment|safety\s+(?:rules|measures|guidelines))"
                                r"|remove\s+(?:(?:the|all|any|its|my|these)\s+){0,3}(?:(?:content|safety)\s+"
                                r"(?:filters?|filtering)|guardrails|safeguards|censorship|alignment"
                                r"|safety\s+(?:rules|measures|guidelines)))\b")),
    ("en_disable_your_safety", 4, _r(r"\b(?:bypass|disable|turn\s+off|remove|deactivate|circumvent|evade|get\s+around"
                                     r"|break|ignore)\s+your\s+(?:\w+\s+){0,2}(?:filters?|restrictions|guardrails"
                                     r"|safeguards|censorship|policies|guidelines|alignment|safety|programming"
                                     r"|limitations)\b")),
    ("en_new_instructions", 1, _r(r"\b(?:new|updated|real|actual)\s+(?:instructions|system\s+prompt|rules)\b")),
    ("en_prompt_injection_mention", 1, _r(r"\bprompt\s+injection\b")),
    # --- Turkish: override previous instructions
    ("tr_ignore_instructions", 4, _r(rf"\b(?:{_TR_QUAL}\s+){{0,3}}{_TR_STRONG_NOUN}\s+(?:\w+\s+){{0,2}}{_TR_VERB}")),
    ("tr_ignore_instructions_vo", 4, _r(rf"\b{_TR_VERB}\s+(?:{_TR_QUAL}\s+){{0,2}}{_TR_STRONG_NOUN}")),
    ("tr_ignore_rules_qualified", 4, _r(rf"\b{_TR_QUAL}\s+(?:\w+\s+)?(?:(?:kural|filtre|kisitlama|sinirlama"
                                        rf"|etik\s+kural|guvenlik\s+onlem)\w*\s+(?:\w+\s+){{0,2}}{_TR_VERB}"
                                        rf"|komut\w*\s+(?:\w+\s+){{0,2}}{_TR_IGNORE_VERB})")),
    ("tr_ignore_rules", 2, _r(rf"\b(?:kural\w*\s+(?:\w+\s+){{0,1}}{_TR_VERB}"
                              rf"|komut\w*\s+(?:\w+\s+){{0,1}}{_TR_IGNORE_VERB})")),
    ("tr_ignore_everything", 2, _r(rf"\b(?:her\s+seyi|hersey|her\s+sey|tum\s+bunlari|bunlari|oncekileri)\s+{_TR_VERB}")),
    ("tr_ignore_above", 3, _r(rf"\b(?:yukaridakileri|yukaridaki\s+her\s+seyi|yukarida\s+yazanlari|onceki\s+mesajlari)"
                              rf"\s+{_TR_VERB}")),
    # --- Turkish: prompt extraction
    ("tr_reveal_prompt", 4, _r(rf"\b(?:{_TR_OWN_QUAL}\s+(?:{_TR_WEAK_QUAL}\s+)?{_TR_INSTR_NOUN}"
                               rf"|{_TR_WEAK_QUAL}\s+{_TR_PROMPT_NOUN}"
                               rf"|{_TR_SECOND_PERSON}\s+(?:\w+\s+){{0,2}}{_TR_WEAK_QUAL}\s+{_TR_INSTR_NOUN})"
                               rf"\s+(?:\w+\s+){{0,2}}{_TR_REVEAL_VERB}")),
    ("tr_reveal_prompt_ov", 4, _r(rf"\b(?:goster|yaz|soyle|paylas|tekrarla|listele|yazdir|ifsa\s+et)\w*\s+"
                                  rf"(?:bana\s+)?(?:{_TR_OWN_QUAL}\s+(?:{_TR_WEAK_QUAL}\s+)?(?:prompt|istem|talimat"
                                  rf"|yonerge)\w*|{_TR_WEAK_QUAL}\s+{_TR_PROMPT_NOUN})")),
    ("tr_reveal_system_message", 2, _r(rf"\bsistem\s+mesaj\w*\s+(?:\w+\s+){{0,2}}{_TR_REVEAL_VERB}")),
    ("tr_given_instructions", 3, _r(r"\b(?:sana|size)\s+(?:verilen|yazilan|tanimlanan|yuklenen|ogretilen)\s+"
                                    r"(?:talimat|komut|prompt|kural|yonerge|istem|direktif)\w*")),
    ("tr_system_prompt_mention", 2, _r(r"\bsistem\s+(?:prompt|istem)\w*")),
    # --- Turkish: persona / mode switching
    ("tr_you_are_not", 3, _r(r"\b(?:artik\s+sen|sen\s+artik)\s+(?:\w+\s+){0,3}degilsin\b")),
    ("tr_bad_persona", 4, _r(rf"\b(?:artik|bundan\s+sonra|su\s+andan\s+itibaren|simdiden\s+sonra|sen)\s+"
                             rf"(?:sen\s+)?(?:bir\s+)?(?:\w+\s+){{0,2}}{_TR_BAD_ADJ}\s*(?:bir\s+)?"
                             rf"(?:yapay\s+zeka|ai|asistan|bot|model|chatbot|sohbet\s+botu|mod|karakter|ol|sin|sun"
                             rf"|olacaksin|davran)\w*")),
    ("tr_bad_ai", 4, _r(rf"\b{_TR_BAD_ADJ}\s+(?:bir\s+)?(?:yapay\s+zeka|ai|asistan|bot|model|chatbot|sohbet\s+botu)\w*")),
    ("tr_from_now_you", 2, _r(r"\b(?:artik|bundan\s+sonra|su\s+andan\s+itibaren)\s+sen\b")),
    ("tr_bad_mode", 4, _r(r"\b(?:kisitlamasiz|sansursuz|filtresiz|kuralsiz|sinirsiz|limitsiz|jailbreak|dan|serbest)\s+"
                          r"mod(?:u|a|da|unda|unu|una)?\b")),
    ("tr_developer_mode", 1, _r(r"\b(?:gelistirici|tanri|admin|yonetici)\s+mod(?:u|a|da|unda|unu|una)?\b")),
    ("tr_uncensored", 2, _r(r"\b(?:sansursuz|kisitlamasiz|filtresiz|kuralsiz)\b")),
    ("tr_roleplay", 1, _r(r"\brol\s+(?:yap|oyna|yapalim|oynayalim)\w*|\brolune\s+gir\w*|\bgibi\s+davran\w*")),
    ("tr_remove_filters", 2, _r(r"\b(?:filtre|kisitlama|sinirlama|sansur|guvenlik\s+onlem)\w*\s+(?:\w+\s+)?"
                                r"(?:kaldir|devre\s+disi\s+birak|kapat|atla|bypass|by\s+pass|as|asarak)\w*")),
    ("tr_do_anything", 2, _r(r"\bher\s+seyi\s+yapabilen\b")),
]

# High-signal phrases checked against the compact view (all separators
# removed) to catch "i.g.n.o.r.e   p r e v i o u s" style obfuscation.
_COMPACT_RULES: List[Tuple[str, int, str]] = [
    ("en_ignore_instructions", 4, "ignorepreviousinstructions"),
    ("en_ignore_instructions", 4, "ignoreallpreviousinstructions"),
    ("en_ignore_instructions", 4, "ignoreallinstructions"),
    ("en_ignore_instructions", 4, "disregardpreviousinstructions"),
    ("en_ignore_instructions", 4, "disregardallpreviousinstructions"),
    ("en_ignore_instructions", 4, "ignoreyourinstructions"),
    ("en_reveal_system_prompt", 4, "revealyoursystemprompt"),
    ("en_reveal_system_prompt", 4, "showmeyoursystemprompt"),
    ("en_reveal_system_prompt", 4, "printyoursystemprompt"),
    ("en_jailbreak", 4, "jailbreak"),
    ("en_do_anything_now", 4, "doanythingnow"),
    ("tr_ignore_instructions", 4, "oncekitalimatlariunut"),
    ("tr_ignore_instructions", 4, "tumtalimatlariunut"),
    ("tr_ignore_instructions", 4, "talimatlariunut"),
    ("tr_ignore_instructions", 4, "talimatlariyoksay"),
    ("tr_ignore_instructions", 4, "talimatlarigormezdengel"),
    ("tr_reveal_prompt", 4, "sistempromptunugoster"),
    ("tr_reveal_prompt", 4, "sistemtalimatlarinigoster"),
]

_UPPER_DAN_RE = re.compile(r"(?<!['’\w])DAN(?![\w'’])")


def score_injection(text: str) -> Tuple[int, List[str]]:
    """Return (score, matched rule names) for a normalised message."""
    hits = {}
    views, compact_views = _matching_views(text)
    for name, weight, regex in _RULES:
        if any(regex.search(view) for view in views):
            hits[name] = max(hits.get(name, 0), weight)
    for name, weight, phrase in _COMPACT_RULES:
        if any(phrase in compact for compact in compact_views):
            hits[name] = max(hits.get(name, 0), weight)
    # Upper-case standalone "DAN" (not Turkish "Ankara'DAN" style suffixes).
    if _UPPER_DAN_RE.search(text) and not text.isupper():
        hits.setdefault("en_dan_upper", 2)
    # Forged turns / special tokens in the raw input are a signal too.
    roles = {fold_text(m.group(1)) for m in _ROLE_MARKER_RE.finditer(text)}
    if roles:
        hits["role_markers"] = min(2, len(roles))
    if _SPECIAL_TOKEN_RE.search(text):
        hits["special_tokens"] = 2
    return sum(hits.values()), sorted(hits)


# ---------------------------------------------------------------------------
# Output leak detection
# ---------------------------------------------------------------------------

_SHINGLE = 6
_LEAK_MIN_SHINGLES = 4
_protected_shingles: set = set()
_protected_texts: List[str] = []


def _shingle_words(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9]+", " ", fold_text(text)).split()


def _shingles(words: List[str]) -> set:
    return {tuple(words[i:i + _SHINGLE]) for i in range(len(words) - _SHINGLE + 1)}


def register_protected_text(*texts: Iterable[str]) -> None:
    """Register texts (system prompts) that must never be echoed verbatim."""
    for text in texts:
        if not text or text in _protected_texts:
            continue
        _protected_texts.append(text)
        _protected_shingles.update(_shingles(_shingle_words(text)))


def contains_protected_text(text: str) -> bool:
    """True if ``text`` repeats a substantial verbatim chunk of a protected text
    (>= 4 shared 6-word shingles, i.e. a run of about 9+ words)."""
    if not _protected_shingles or not text:
        return False
    found = _shingles(_shingle_words(text)) & _protected_shingles
    return len(found) >= _LEAK_MIN_SHINGLES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _preview(text: str) -> str:
    snippet = text[:_PREVIEW_CHARS]
    return repr(snippet + ("…" if len(text) > _PREVIEW_CHARS else ""))


def guard_input(text: str, language: Optional[str] = None) -> GuardResult:
    """Normalise a user message and decide whether it may reach the model."""
    raw = "" if text is None else str(text)
    if not is_enabled():
        return GuardResult(allowed=True, text=raw)

    normalized = normalize_text(raw)
    lang = _lang(language, normalized)

    if not normalized:
        return GuardResult(allowed=False, text="", reason="empty", refusal=_EMPTY[lang])

    limit = _max_chars()
    if len(normalized) > limit:
        logger.warning("Prompt guard blocked input (reason=too_long, length=%d, limit=%d): %s",
                       len(normalized), limit, _preview(normalized))
        return GuardResult(
            allowed=False,
            text=normalized[:limit],
            reason="too_long",
            refusal=_TOO_LONG[lang].format(length=len(normalized), limit=limit),
        )

    score, rules = score_injection(normalized)
    cleaned = neutralize_markers(normalized)
    if score >= BLOCK_THRESHOLD:
        reason = f"prompt_injection(score={score}; {','.join(rules)})"
        logger.warning("Prompt guard blocked input (%s, length=%d): %s", reason, len(normalized),
                       _preview(normalized))
        return GuardResult(allowed=False, text=cleaned, reason=reason, refusal=_REFUSAL[lang])

    if not cleaned:
        return GuardResult(allowed=False, text="", reason="empty", refusal=_EMPTY[lang])
    return GuardResult(allowed=True, text=cleaned)


def is_fallback_answer(text: Optional[str]) -> bool:
    """True if ``text`` is the safe fallback that guard_output() substitutes for a
    leaking/unsafe answer. Callers mark such rows so they never become training data."""
    return bool(text) and text in _OUTPUT_FALLBACK.values()


def guard_output(text: str, language: Optional[str] = None) -> str:
    """Clean a generated answer; replace it with a safe fallback if it leaks the system prompt."""
    raw = "" if text is None else str(text)
    if not is_enabled():
        return raw
    lang = _lang(language, raw)
    cleaned = strip_reasoning(raw)  # before everything else: reasoning is never part of the answer
    cleaned = cut_at_end_of_turn(cleaned)
    cleaned = strip_special_tokens(cleaned)
    cleaned = cut_at_turn_markers(cleaned)
    cleaned = _HSPACE_RE.sub(" ", cleaned).strip()
    if contains_protected_text(cleaned):
        logger.warning("Prompt guard replaced an answer that leaked protected text: %s", _preview(cleaned))
        return _OUTPUT_FALLBACK[lang]
    if not cleaned:
        return _OUTPUT_FALLBACK[lang]
    return cleaned

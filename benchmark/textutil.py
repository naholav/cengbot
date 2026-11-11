"""Text helpers shared by the benchmark builder and grader (stdlib only).

* ``html_to_text``: readable text from an HTML page (scripts, styles, menus,
  headers and footers are dropped; block elements become line breaks).
* ``norm_for_match``: whitespace/case/quote folding used for the verbatim
  evidence check (Turkish dotted/dotless i are folded to ``i``).
* ``fold``: ASCII folding for token comparisons.
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser
from typing import List, Optional

_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "h1", "h2", "h3", "h4",
    "h5", "h6", "dd", "dt", "dl", "blockquote", "pre", "hr", "td", "th", "tbody", "thead", "main",
    "aside", "form", "fieldset", "figure", "figcaption", "caption", "center",
}
_DROP_TAGS = {"script", "style", "noscript", "template", "svg", "head", "iframe", "select", "button"}
_CHROME_TAGS = {"nav", "header", "footer"}
_CHROME_CLASS_RE = re.compile(r"\b(navbar|topbar|web-header|footer[\w-]*|breadcrumb|cookie[\w-]*|skip-link)\b",
                              re.IGNORECASE)
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track",
              "wbr", "param"}


class _TextExtractor(HTMLParser):
    def __init__(self, keep_chrome: bool = False):
        super().__init__(convert_charrefs=True)
        self.keep_chrome = keep_chrome
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self._in_title = False
        self._skip_tag: Optional[str] = None
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if self._skip_tag is not None:
            if tag == self._skip_tag and tag not in _VOID_TAGS:
                self._skip_depth += 1
            return
        attr = dict(attrs)
        chrome = not self.keep_chrome and (
            tag in _CHROME_TAGS or bool(_CHROME_CLASS_RE.search(attr.get("class") or ""))
            or bool(_CHROME_CLASS_RE.search(attr.get("id") or "")))
        if (tag in _DROP_TAGS or chrome) and tag not in _VOID_TAGS:
            self._skip_tag, self._skip_depth = tag, 1
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_startendtag(self, tag, attrs):
        if self._skip_tag is None and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip_tag = None
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        if self._skip_tag is None:
            self.parts.append(data)


def clean_lines(text: str) -> str:
    text = text.replace("\r", "\n").replace("\xa0", " ").replace("​", "")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    out: List[str] = []
    for line in lines:
        if line.strip(" |"):
            out.append(line.strip())
        elif out and out[-1] != "":
            out.append("")
    return "\n".join(out).strip()


def html_to_text(html: str, keep_chrome: bool = False) -> tuple:
    """Return (title, text) for an HTML document."""
    parser = _TextExtractor(keep_chrome=keep_chrome)
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: keep what was parsed
        pass
    title = " ".join("".join(parser.title_parts).split())
    return title, clean_lines("".join(parser.parts))


_QUOTES = str.maketrans({"’": "'", "‘": "'", "‛": "'", "´": "'", "`": "'", "“": '"', "”": '"', "„": '"',
                         "–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-", "\xa0": " ", "…": "..."})
_I_FOLD = str.maketrans({"İ": "i", "I": "i", "ı": "i"})


def norm_for_match(text: str) -> str:
    """Normalise text for a case/whitespace-insensitive verbatim comparison."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_QUOTES).translate(_I_FOLD).lower()
    text = text.replace("i̇", "i")  # 'İ'.lower() leftovers
    text = text.replace("|", " ")
    return " ".join(text.split())


_TR_ASCII = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "â": "a", "î": "i",
                           "û": "u", "Ç": "c", "Ğ": "g", "İ": "i", "I": "i", "Ö": "o", "Ş": "s", "Ü": "u"})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lowercase ASCII fold (Turkish letters mapped to their ASCII base)."""
    text = unicodedata.normalize("NFKC", text or "").translate(_TR_ASCII).lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(fold(text))

#!/usr/bin/env python3
"""Build / refresh the benchmark question set from official university pages.

Pipeline (each step is a sub-command; downloads are untrusted data, so run the
steps that read them with ``python -I``):

  fetch     polite download (<= 1 request/s, ordinary browser user agent) of the
            URLs in ``--url-file`` (one per line, ``#`` comments) into
            ``--raw-dir``; writes ``manifest.jsonl`` (url, final_url, status,
            content_type, fetched_at, file).
  extract   HTML -> text (stdlib parser) and PDF -> text (``pdftotext``) into
            ``--text-dir``; writes ``pages.jsonl`` (url, final_url, title,
            fetched_at, text_file, chars).
  generate  grounded candidate generation with the Claude API: for every page
            the extracted text is sent and the model is asked for Q&A items that
            can be answered ONLY from that text (question, reference_answer,
            key_facts, required link, verbatim evidence). Needs
            ``ANTHROPIC_API_KEY``/``CLAUDE_API_KEY`` in ``.env``.
  verify    automatic checks for every candidate (evidence quote found verbatim
            in the page text, required links answer HTTP 200, question not a
            near-duplicate of a training/DB question) + the manual review
            decisions in ``--review`` (JSON: id -> {"ok": bool, "reason": str}),
            then writes the final ``questions.jsonl`` (``--keep`` items, balanced
            over language and category) and a ``build_report.json``.

Default cache directory: ``benchmark/.cache`` (git-ignored).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # also works under ``python -I``
    sys.path.insert(0, str(_HERE))

import textutil  # noqa: E402

ROOT = _HERE.parent
DEFAULT_CACHE = _HERE / ".cache"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0 Safari/537.36")
MIN_INTERVAL_S = 1.1
SIMILARITY_THRESHOLD = 0.8


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_url_file(path: Path) -> List[str]:
    urls = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line.split()[0])  # anything after the URL is a comment
    return list(dict.fromkeys(urls))


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

class PoliteSession:
    """requests.Session with a minimum interval between requests."""

    def __init__(self, min_interval: float = MIN_INTERVAL_S, timeout: float = 30.0):
        import requests

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT,
                                     "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8"})
        self.min_interval = min_interval
        self.timeout = timeout
        self._last = 0.0

    def _wait(self) -> None:
        delay = self.min_interval - (time.monotonic() - self._last)
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()

    def get(self, url: str, **kwargs):
        self._wait()
        return self.session.get(url, timeout=self.timeout, allow_redirects=True, **kwargs)

    def head_ok(self, url: str) -> Tuple[bool, Optional[int], Optional[str]]:
        """True if ``url`` answers 200 (after redirects). Uses a streamed GET: some servers
        (e.g. the EBS) answer HEAD with 404 although GET works; the body is not downloaded."""
        import requests

        try:
            self._wait()
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True, stream=True)
            resp.close()
            return resp.status_code == 200, resp.status_code, resp.url
        except requests.RequestException as exc:
            return False, None, f"{type(exc).__name__}"


def page_key(url: str, variant: str = "") -> str:
    return url_key(f"{url}|{variant}" if variant else url)


def cmd_fetch(args) -> int:
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.jsonl"
    variant = args.variant or ""
    rows = read_jsonl(manifest_path) if manifest_path.exists() else []
    existing = {(row["url"], row.get("variant", "")): row for row in rows}
    urls = (read_url_file(Path(args.url_file)) if args.url_file else []) + list(args.urls or [])
    session = PoliteSession()
    if args.warmup_url:  # e.g. https://bmb.cu.edu.tr/lang/en (the site keeps the language in the session)
        session.get(args.warmup_url)
    for url in urls:
        old = existing.get((url, variant))
        if old and old.get("status") == 200 and not args.refresh:
            continue
        row: Dict[str, Any] = {"url": url, "variant": variant, "fetched_at": _now_iso()}
        try:
            resp = session.get(url)
            row.update(status=resp.status_code, final_url=resp.url,
                       content_type=resp.headers.get("Content-Type", ""))
            if resp.status_code == 200:
                ctype = row["content_type"].lower()
                ext = ".pdf" if ("pdf" in ctype or resp.content[:5] == b"%PDF-") else ".html"
                name = page_key(url, variant) + ext
                (raw_dir / name).write_bytes(resp.content)
                row["file"] = name
                row["bytes"] = len(resp.content)
        except Exception as exc:  # network errors are recorded, not fatal
            row.update(status=None, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        existing[(url, variant)] = row
        print(f"{row.get('status')}\t{variant or '-'}\t{url}", flush=True)
        write_jsonl(manifest_path, existing.values())
    return 0


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def _decode_html(data: bytes) -> str:
    head = data[:4096].decode("ascii", errors="ignore").lower()
    match = re.search(r"charset=[\"']?([\w-]+)", head)
    for enc in ([match.group(1)] if match else []) + ["utf-8", "windows-1254", "iso-8859-9"]:
        try:
            return data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def pdf_to_text(path: Path) -> str:
    try:
        out = subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"], capture_output=True,
                             timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return textutil.clean_lines(out.stdout.decode("utf-8", errors="replace"))


def cmd_extract(args) -> int:
    raw_dir, text_dir = Path(args.raw_dir), Path(args.text_dir)
    text_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for row in read_jsonl(raw_dir / "manifest.jsonl"):
        if row.get("status") != 200 or not row.get("file"):
            continue
        src = raw_dir / row["file"]
        if src.suffix == ".pdf":
            title, text = Path(row["url"]).name, pdf_to_text(src)
        else:
            title, text = textutil.html_to_text(_decode_html(src.read_bytes()))
        name = page_key(row["url"], row.get("variant", "")) + ".txt"
        (text_dir / name).write_text(text, encoding="utf-8")
        pages.append({"url": row["url"], "variant": row.get("variant", ""), "final_url": row.get("final_url"),
                      "title": title, "fetched_at": row.get("fetched_at"), "text_file": name,
                      "chars": len(text)})
    write_jsonl(text_dir / "pages.jsonl", pages)
    print(f"extracted {len(pages)} pages into {text_dir}")
    return 0


def load_pages(text_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """url -> list of page versions (one per language variant), each with its ``text``."""
    pages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(Path(text_dir) / "pages.jsonl"):
        row["text"] = (Path(text_dir) / row["text_file"]).read_text(encoding="utf-8")
        pages[row["url"]].append(row)
        final = row.get("final_url")
        if final and final != row["url"]:
            pages[final].append(row)
    return pages


# ---------------------------------------------------------------------------
# generate (Claude API, grounded)
# ---------------------------------------------------------------------------

GEN_MODEL = "claude-sonnet-5-5"
CATEGORIES = [
    "academic_calendar", "registration", "internship", "erasmus_exchange", "graduation", "staff_contacts",
    "courses_credits", "labs_research", "double_major_minor", "exams", "student_affairs", "department_general",
    "graduate_programs",
]

GEN_SYSTEM = """You write evaluation items for a small assistant that answers students' questions about \
Çukurova University's Computer Engineering department (Bilgisayar Mühendisliği). The assistant answers in \
the language of the question, in one to three short sentences, and includes the official link where the \
information lives.

You receive the extracted text of ONE official web page and its URL. Write factual question-answer items \
that can be answered ONLY from this page text. Rules:
- Every fact in reference_answer and key_facts must be stated explicitly in the page text. Do not use \
outside knowledge; do not infer.
- evidence: copy one contiguous passage (10-60 words) VERBATIM from the page text that proves the answer. \
Do not fix typos or change punctuation inside it.
- Prefer stable facts (rules, requirements, durations, credits, course codes, contact units, which office \
handles what). Skip news, names of people in temporary roles, and anything that will change soon. If a fact \
is tied to a date (academic calendar, deadlines), put the academic year in the question.
- Questions must sound like a real student and must be answerable without seeing the page.
- reference_answer: concise (1-3 sentences) in the question's language, ending with the URL.
- key_facts: 2-4 short atomic facts a correct answer must contain (in the question's language).
- difficulty: easy (single fact), medium (two facts or a condition), hard (combines several facts or a \
precise number/condition).
Return ONLY a JSON object: {"items": [{"lang": "tr"|"en", "category": one of %s, "difficulty": \
"easy"|"medium"|"hard", "question": str, "reference_answer": str, "key_facts": [str], "evidence": str}]}. \
If the page has no usable facts return {"items": []}.""" % json.dumps(CATEGORIES)


def cmd_generate(args) -> int:  # pragma: no cover - needs the network and an API key
    import llm_api

    client = llm_api.make_client()
    if client is None:
        print("No ANTHROPIC_API_KEY / CLAUDE_API_KEY in .env; generation skipped.", file=sys.stderr)
        return 2
    ok, why = llm_api.check_access(client, GEN_MODEL)
    if not ok:
        print(f"The Claude API cannot be used: {why}. Generation skipped.", file=sys.stderr)
        return 2
    pages = {url: versions[0] for url, versions in load_pages(Path(args.text_dir)).items()
             if versions and versions[0]["url"] == url}
    selected = [u for u in pages if not args.pages or u in set(args.pages)]
    out_path = Path(args.out)
    rows = read_jsonl(out_path) if out_path.exists() else []
    done = {row["source_url"] for row in rows}
    usage: Counter = Counter()
    for url in selected:
        if url in done:
            continue
        page = pages[url]
        text = page["text"][: args.max_chars]
        user = (f"URL: {url}\nTitle: {page.get('title', '')}\n\nPAGE TEXT:\n<<<\n{text}\n>>>\n\n"
                f"Write up to {args.per_page} items ({args.lang_hint}).")
        try:
            data, info = llm_api.call_json(client, model=GEN_MODEL, system=GEN_SYSTEM, user=user,
                                           max_tokens=16000, effort="medium")
        except llm_api.LLMFatalError as exc:
            print(f"stopping: {exc}", file=sys.stderr)
            break
        except llm_api.LLMError as exc:
            usage.update(exc.info.get("usage", {}))
            print(f"skip {url}: {exc} ({exc.status})", file=sys.stderr)
            continue
        usage.update(info.get("usage", {}))
        items = data.get("items", []) if isinstance(data, dict) else []
        for n, item in enumerate(items):
            if not isinstance(item, dict) or not item.get("question"):
                continue
            item = dict(item)
            item["cand_id"] = f"g{url_key(url)}-{n:02d}"
            item["source_url"] = url
            item["required_links"] = [url]
            item["acceptable_link_prefixes"] = [url]
            item["origin"] = f"generated:{GEN_MODEL}"
            rows.append(item)
        write_jsonl(out_path, rows)
        print(f"{url}: {len(rows)} candidates so far", flush=True)
    cost = llm_api.estimate_cost(GEN_MODEL, dict(usage))
    print(f"done: {len(rows)} candidates, usage={dict(usage)}, estimated cost ${cost:.2f}")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def load_reference_questions(training_file: Optional[Path], db_path: Optional[Path]) -> List[str]:
    questions: List[str] = []
    if training_file and Path(training_file).exists():
        for row in read_jsonl(Path(training_file)):
            if isinstance(row.get("question"), str):
                questions.append(row["question"])
    if db_path and Path(db_path).exists():
        import sqlite3

        uri = f"file:{Path(db_path).resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            questions.extend(q for (q,) in conn.execute("SELECT question FROM raw_data") if isinstance(q, str))
        finally:
            conn.close()
    return questions


class SimilarityIndex:
    """Max character-level (difflib) and TF-IDF cosine similarity against reference questions."""

    def __init__(self, questions: Sequence[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.questions = [q for q in questions if q and q.strip()]
        self.folded = [textutil.fold(q) for q in self.questions]
        self.word = TfidfVectorizer(analyzer="word", token_pattern=r"[a-z0-9]+", sublinear_tf=True)
        self.word_matrix = self.word.fit_transform(self.folded)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.char_matrix = self.char.fit_transform(self.folded)

    def max_similarity(self, question: str) -> Dict[str, Any]:
        import difflib

        folded = textutil.fold(question)
        word_sims = (self.word_matrix @ self.word.transform([folded]).T).toarray().ravel()
        char_sims = (self.char_matrix @ self.char.transform([folded]).T).toarray().ravel()
        best_word, best_char = int(word_sims.argmax()), int(char_sims.argmax())
        # exact character ratio on the closest few candidates (difflib is slow on 20k strings)
        top = set(word_sims.argsort()[-5:]) | set(char_sims.argsort()[-5:])
        ratio, ratio_idx = 0.0, best_char
        for idx in top:
            r = difflib.SequenceMatcher(None, folded, self.folded[idx]).ratio()
            if r > ratio:
                ratio, ratio_idx = r, idx
        best = max(float(word_sims[best_word]), float(char_sims[best_char]), ratio)
        nearest = {float(word_sims[best_word]): best_word, float(char_sims[best_char]): best_char,
                   ratio: ratio_idx}[best]
        return {"max": round(best, 3), "tfidf_word": round(float(word_sims[best_word]), 3),
                "tfidf_char": round(float(char_sims[best_char]), 3), "char_ratio": round(ratio, 3),
                "nearest": self.questions[nearest]}


def quote_found(evidence: str, page_text: str) -> bool:
    ev = textutil.norm_for_match(evidence)
    return bool(ev) and ev in textutil.norm_for_match(page_text)


def _balanced_pick(items: List[Dict[str, Any]], keep: int) -> List[Dict[str, Any]]:
    """Round-robin over (lang, category) so both languages and all categories are represented."""
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[(item["lang"], item["category"])].append(item)
    per_lang = {"tr": keep // 2, "en": keep - keep // 2}
    picked: List[Dict[str, Any]] = []
    for lang in ("tr", "en"):
        keys = sorted(k for k in buckets if k[0] == lang)
        queues = {k: list(buckets[k]) for k in keys}
        chosen: List[Dict[str, Any]] = []
        while len(chosen) < per_lang[lang] and any(queues.values()):
            for k in keys:
                if queues[k] and len(chosen) < per_lang[lang]:
                    chosen.append(queues[k].pop(0))
        picked.extend(chosen)
    if len(picked) < keep:  # one language ran short: fill from the rest
        rest = [i for i in items if i not in picked]
        picked.extend(rest[: keep - len(picked)])
    return picked


def _link_prefixes(url: str) -> List[str]:
    return [url.rstrip("/")]


def cmd_verify(args) -> int:
    pages = load_pages(Path(args.text_dir))
    candidates = read_jsonl(Path(args.candidates))
    review = json.loads(Path(args.review).read_text(encoding="utf-8")) if args.review else {}
    index = SimilarityIndex(load_reference_questions(
        Path(args.training_file) if args.training_file else None, Path(args.db) if args.db else None))
    session = PoliteSession() if args.check_links else None
    link_cache: Dict[str, Tuple[bool, Optional[int], Optional[str]]] = {}
    seen_questions: List[str] = []
    kept, dropped = [], []
    for cand in candidates:
        cid = cand.get("cand_id") or cand.get("id")
        reasons = []
        versions = pages.get(cand.get("source_url", "")) or []
        if not versions:
            reasons.append("source page not fetched")
        found = any(quote_found(cand.get("evidence", ""), v["text"]) for v in versions)
        if versions and not found:
            reasons.append("evidence quote not found verbatim")
        links_ok = True
        link_status = {}
        if session is not None:
            for link in dict.fromkeys(list(cand.get("required_links", [])) + [cand.get("source_url", "")]):
                if not link:
                    continue
                if link not in link_cache:
                    link_cache[link] = session.head_ok(link)
                ok, status, final = link_cache[link]
                link_status[link] = status
                links_ok = links_ok and ok
            if not links_ok:
                reasons.append(f"required link not HTTP 200: {link_status}")
        else:
            links_ok = bool(cand.get("checks", {}).get("link_ok"))
            if not links_ok:
                reasons.append("links not checked (run with --check-links)")
        sim = index.max_similarity(cand["question"])
        if sim["max"] >= SIMILARITY_THRESHOLD:
            reasons.append(f"too similar to an existing question ({sim['max']}): {sim['nearest'][:80]!r}")
        for other in seen_questions:
            import difflib

            if difflib.SequenceMatcher(None, textutil.fold(cand["question"]), textutil.fold(other)).ratio() >= 0.9:
                reasons.append("duplicate of another candidate")
                break
        decision = review.get(cid, {})
        manual_ok = bool(decision.get("ok", False))
        if not manual_ok:
            reasons.append("manual review: " + (decision.get("reason") or "not approved"))
        record = dict(cand)
        record["checks"] = {"quote_found": found, "link_ok": links_ok, "manual_ok": manual_ok,
                            "max_train_similarity": sim["max"]}
        if reasons:
            dropped.append({"cand_id": cid, "question": cand.get("question"), "reasons": reasons})
        else:
            record["_select"] = decision.get("select", True) is not False
            kept.append(record)
            seen_questions.append(cand["question"])
    # reviewer-marked reserves pass every check but are only used if the selection runs short
    eligible = [k for k in kept if k["_select"]]
    picked = _balanced_pick(eligible, args.keep)
    if len(picked) < args.keep:
        picked += _balanced_pick([k for k in kept if not k["_select"]], args.keep - len(picked))
    counters = Counter()
    final = []
    id_map: Dict[str, Any] = {}
    for item in picked:
        counters[item["lang"]] += 1
        row = {
            "id": f"{item['lang']}-{counters[item['lang']]:03d}",
            "lang": item["lang"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "key_facts": item["key_facts"],
            **({"extra_facts": item["extra_facts"]} if item.get("extra_facts") else {}),  # unscored context
            "required_links": item["required_links"],
            "acceptable_link_prefixes": item.get("acceptable_link_prefixes")
            or sorted({p for link in item["required_links"] for p in _link_prefixes(link)}),
            "source_url": item["source_url"],
            "evidence": item["evidence"],
            "verified_at": args.verified_at,
            "checks": {k: item["checks"][k] for k in ("quote_found", "link_ok", "manual_ok")},
            "origin": item.get("origin", "manual"),
        }
        final.append(row)
        id_map[row["id"]] = item.get("cand_id")
    write_jsonl(Path(args.out), final)
    report = {
        "candidates": len(candidates), "passed": len(kept), "kept": len(final),
        "not_selected": len(kept) - len(final), "dropped": dropped,
        "reserve": [{"cand_id": k.get("cand_id"), "question": k["question"]} for k in kept if k not in picked],
        "id_map": id_map,
        "kept_by_lang": dict(Counter(r["lang"] for r in final)),
        "kept_by_category": dict(Counter(r["category"] for r in final)),
        "kept_by_difficulty": dict(Counter(r["difficulty"] for r in final)),
        "link_status": {k: v[1] for k, v in link_cache.items()},
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("dropped", "link_status", "reserve", "id_map")},
                     ensure_ascii=False))
    for d in dropped:
        print(f"DROP {d['cand_id']}: {'; '.join(d['reasons'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--url-file", default=str(_HERE / "source_urls.txt"))
    f.add_argument("--raw-dir", default=str(DEFAULT_CACHE / "raw"))
    f.add_argument("--variant", default="", help="label for a language variant, e.g. 'en'")
    f.add_argument("--warmup-url", help="URL requested first in the same session (language switch)")
    f.add_argument("--refresh", action="store_true")
    f.add_argument("urls", nargs="*")
    f.set_defaults(func=cmd_fetch)

    e = sub.add_parser("extract")
    e.add_argument("--raw-dir", default=str(DEFAULT_CACHE / "raw"))
    e.add_argument("--text-dir", default=str(DEFAULT_CACHE / "text"))
    e.set_defaults(func=cmd_extract)

    g = sub.add_parser("generate")
    g.add_argument("--text-dir", default=str(DEFAULT_CACHE / "text"))
    g.add_argument("--out", default=str(DEFAULT_CACHE / "candidates.jsonl"))
    g.add_argument("--pages", nargs="*")
    g.add_argument("--per-page", type=int, default=6)
    g.add_argument("--max-chars", type=int, default=60000)
    g.add_argument("--lang-hint", default="half Turkish, half English")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("verify")
    v.add_argument("--candidates", default=str(DEFAULT_CACHE / "candidates.jsonl"))
    v.add_argument("--text-dir", default=str(DEFAULT_CACHE / "text"))
    v.add_argument("--review", help="JSON file: cand_id -> {ok: bool, reason: str}")
    v.add_argument("--training-file", default=str(ROOT / "data" / "merged_qa_data.jsonl"))
    v.add_argument("--db", default=str(ROOT / "university_bot.db"))
    v.add_argument("--check-links", action="store_true")
    v.add_argument("--keep", type=int, default=100)
    v.add_argument("--verified-at", default=_dt.date.today().isoformat())
    v.add_argument("--out", default=str(_HERE / "questions.jsonl"))
    v.add_argument("--report", default=str(DEFAULT_CACHE / "build_report.json"))
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

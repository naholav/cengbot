# CengBot benchmark

A fixed, verified question set plus a grader, so that every model (LLaMA-3.2-3B + LoRA today; Qwen, Gemma,
Mistral, ... after retraining) is scored the same way and can be compared on
[`LEADERBOARD.md`](LEADERBOARD.md).

The benchmark evaluates **the model alone** (the bot is fine-tuned, there is no retrieval): each answer is
generated through the production path and compared with a reference answer that was checked against the
official page.

## Contents

| file | what it is |
|---|---|
| `questions.jsonl` | the 100 benchmark items (50 Turkish, 50 English) |
| `run_benchmark.py` | generates answers with a model, grades them, writes a run directory, refreshes the leaderboard |
| `grader.py` | deterministic checks, LLM judge, scoring and aggregation |
| `llm_api.py` | Anthropic API wrapper (key handling, typed retries, stop-reason checks, robust JSON parsing) |
| `build_questions.py` | builder: `fetch` / `extract` / `generate` / `verify` |
| `textutil.py` | HTML-to-text and text normalisation helpers |
| `sources.md`, `source_urls*.txt` | the official pages used, and when they were fetched |
| `provenance/` | all 151 candidates, the review decisions and the build report (why each one was kept or dropped) |
| `results/<run>/` | one directory per run (committed: small text files) |
| `LEADERBOARD.md` | generated table of all full runs |

Raw page downloads go to `benchmark/.cache/` (git-ignored) and are never committed.

## The question set

Each line of `questions.jsonl` is one item:

```json
{"id": "tr-001", "lang": "tr", "category": "academic_calendar", "difficulty": "easy",
 "question": "...", "reference_answer": "... https://...", "key_facts": ["...", "..."],
 "required_links": ["https://..."], "acceptable_link_prefixes": ["https://...", "..."],
 "source_url": "https://...", "evidence": "verbatim quote from the page",
 "verified_at": "2026-11-11", "checks": {"quote_found": true, "link_ok": true, "manual_ok": true},
 "origin": "manual"}
```

Categories: academic calendar, registration, internship, Erasmus/exchange, graduation (incl. graduation
project), staff and contacts, courses and credits, labs and research, double major/minor, exams and grading,
student affairs (summer school, leave, student affairs office), graduate programmes, department/accreditation.
Difficulty: 22 easy, 58 medium, 20 hard.

### How the items were made

1. **Sources.** About 70 official pages and PDFs were fetched on 2026-11-11 at ≤ 1 request/s
   (`bmb.cu.edu.tr` in Turkish and English, the EBS course catalogue of programme 555, the 2026-2027 academic
   calendar, the current undergraduate exam regulation, the double major/minor, registration and summer-school
   directives, the department internship directive, Erasmus pages, ...). See `sources.md`. Text was extracted
   with `build_questions.py extract` (run with `python -I`, downloads are treated as untrusted data).
2. **Writing.** The builder can generate grounded candidates with `claude-sonnet-5-5`
   (`build_questions.py generate`: the page text is sent and the model must return items answerable only
   from that text, with a verbatim evidence quote). **For this first version the API key in `.env` was
   rejected (HTTP 401), so the 151 candidates in `provenance/candidates.jsonl` were written by hand from the
   extracted page text**, in the same format and through the same verification. Style and topics follow
   `data/merged_qa_data.jsonl`, but the questions are new.
3. **Verification, item by item** (`build_questions.py verify`):
   * the evidence quote must appear verbatim in the fetched page text (whitespace, case, quotes and
     Turkish dotted/dotless i normalised) — automatic;
   * every required link and the source URL must answer HTTP 200 now (streamed GET; the EBS answers HEAD
     with 404) — automatic;
   * no question may be ≥ 0.8 similar (character ratio, word or character TF-IDF cosine) to any question in
     `data/merged_qa_data.jsonl` or the database's `raw_data.question` (opened read-only) — automatic;
   * manual review against the page text: ambiguous, contradictory or off-topic items were dropped,
     date-bound items name the academic year, volatile facts were avoided where possible.
   Result: 151 candidates, 146 passed, 5 dropped (3 too similar to training questions, 1 where two official
   pages disagree, 1 off-topic), 46 kept in reserve, 100 selected (balanced by language and category).
   Every reference answer is also checked to score 100 on the deterministic grader (`tests/test_benchmark.py`).

## Running

GPU jobs must hold the shared GPU lock (`flock <lock file> <command>`).

```bash
# the production model (models/active-model), LLM judge if a valid API key is configured
flock /path/to/gpu.lock .venv/bin/python benchmark/run_benchmark.py

# a new model after switching it in with scripts/switch_model.sh switch ...
flock /path/to/gpu.lock .venv/bin/python benchmark/run_benchmark.py --label qwen2.5-3b-v2

# or without switching: point the handler at any model directory (sets MODEL_PATH)
flock /path/to/gpu.lock .venv/bin/python benchmark/run_benchmark.py --model-path models/<dir> --label <name>

# re-grade an existing run (no GPU; cached judgments are free), e.g. after adding an API key
.venv/bin/python benchmark/run_benchmark.py --regrade benchmark/results/<run>

# options
--no-llm            deterministic scoring only (no API calls)
--limit N / --ids   partial run (smoke tests; partial runs are not put on the leaderboard)
--seed N            per-item seed (default 1234); the production sampling settings are kept
--judge-effort      low | medium (default) | high
```

The model is loaded offline (`HF_HUB_OFFLINE=1`). Answers go through exactly what the worker does:
`detect_language` → `prompt_guard.guard_input` (blocked prompts get the refusal and never reach the model) →
`generate_response_with_info(..., raise_errors=True, language=...)` → `prompt_guard.guard_output`. Generation
uses the production settings from `.env` (sampling with temperature 0.7); each item is seeded (`seed + index`)
so a rerun on the same machine and library versions is reproducible.

A run writes `benchmark/results/<label-or-model-version>_<timestamp>/`:

* `answers.jsonl` – question, answer, status (`ok` / `blocked` / `output_fallback` / `error`), latency, tokens;
* `judgments.jsonl` – LLM judgments (also the cache; only written when the judge runs);
* `scores.jsonl` – per-item checks and scores;
* `summary.json` – model info (`get_model_info()` + generation settings), git commit and dirty flag, question
  set hash, judge model, rubric version, token usage and cost, all scores;
* `report.md` – the score table and the 10 worst items with answer and reference.

and rebuilds `LEADERBOARD.md` from every `results/*/summary.json`. Commit the run directory and the
leaderboard.

## Scoring

Deterministic checks (always run):

* **Link**: 1 if the answer contains a required link or a URL starting with an acceptable prefix
  (scheme, `www.`, trailing slash, percent-encoding and case are ignored), minus 0.5 if it also contains a
  wrong official URL (a `cu.edu.tr`/official URL that is not on the item's list) or an invented URL, floored
  at 0. *Link accuracy* = share of answers with a correct link.
* **Hallucinated URL**: a URL on a non-official domain that is not on the item's list.
* **Language**: Turkish question → Turkish answer (URLs, e-mails, names and short answers are not judged).
* **Length sanity**: non-empty, not the guard/fallback text, ≤ 150 words.
* **Key-fact coverage (proxy, `det-v2`)**: normalised token overlap per key fact (ASCII folding, stems for
  Turkish suffixes, number words). Only the fact's tokens that are *not already in the question* count, so
  echoing the question earns nothing; every number of the fact must appear, next to the same unit word when
  the fact has one (`2. hafta` is not matched by `2. ve 4. haftalarda`); e-mail addresses must match
  exactly. A fact counts as covered at ≥ 50 % of those tokens. (`det-v1`, a plain overlap, was replaced after
  a hand check showed it gave full credit to wrong answers that reused the question's words.)

LLM judge (`claude-sonnet-5-5`, rubric `v1` in `grader.py`): gets the question, reference answer, key facts,
required links and the model answer and returns strict JSON `{correctness 0-4, key_facts_covered[],
hallucination, language_ok, rationale}` with explicit anchors for each correctness score. Structured outputs
are requested via `output_config.format`; if the API rejects the schema the judge falls back to a JSON-only
prompt and robust parsing.

Per-item score (0-100):

* LLM mode: `50 × correctness/4 + 20 × key-fact coverage + 20 × link + 10 × language`
* deterministic mode (`--no-llm`, or no valid key): `70 × key-fact coverage (token proxy) + 20 × link + 10 × language`
* a hallucination (judge flag, or a hallucinated URL) caps the item at 40.

Reported: overall mean, per language, per category, per difficulty, link accuracy, wrong-link rate,
hallucination rate, language match, length sanity, key-fact coverage, mean correctness.

### Caveats

* **The judge is not deterministic.** Judgments are cached by
  sha256(item id, item text, answer, judge model, rubric version) in the run's `judgments.jsonl`, so a
  re-grade gives the same numbers and costs nothing. Changing the rubric means bumping `RUBRIC_VERSION`
  and re-grading.
* **Only compare like with like:** the same question set (hash in `summary.json`), the same scoring mode,
  and for LLM scores the same judge model and rubric version. The leaderboard keeps LLM-judged and
  deterministic runs in separate tables.
* If the judge refuses or fails on an item, that item falls back to the deterministic formula and the run is
  marked `mixed`; refusals are retried on the next `--regrade`.
* The deterministic proxy cannot tell a correct paraphrase from a wrong answer that reuses the words, and
  it cannot see invented facts: in deterministic mode the *hallucination rate* only counts invented URLs.
  Use it for quick checks, and the LLM judge for decisions.
* Facts are as of 2026-11-11. Calendar items name the 2026-2027 year; staff/role items can go stale.

### Cost per run

The judge sends about 1-1.5 k input tokens per item and returns a short JSON (plus adaptive thinking at
`medium` effort), roughly 0.1-0.15 M input and 0.05-0.15 M output tokens per 100 items: about **$1-2 per full
run** at $2 / $10 per million tokens. Re-grading with cached judgments is free; `summary.json` records the
actual usage and estimated cost.

## Adding or refreshing items when the website changes

```bash
C=benchmark/.cache
.venv/bin/python benchmark/build_questions.py fetch --raw-dir $C/raw --refresh            # source_urls.txt
.venv/bin/python benchmark/build_questions.py fetch --raw-dir $C/raw --refresh \
    --url-file benchmark/source_urls_en.txt --variant en --warmup-url https://bmb.cu.edu.tr/lang/en
.venv/bin/python -I benchmark/build_questions.py extract --raw-dir $C/raw --text-dir $C/text
# optional, needs a valid key: grounded candidates from the page texts
.venv/bin/python -I benchmark/build_questions.py generate --text-dir $C/text --out $C/generated.jsonl
# add/edit candidates in provenance/candidates.jsonl, record decisions in provenance/review.json
#   {"<cand_id>": {"ok": true|false, "select": true|false, "reason": "..."}}
.venv/bin/python -I benchmark/build_questions.py verify --text-dir $C/text --check-links \
    --candidates benchmark/provenance/candidates.jsonl --review benchmark/provenance/review.json \
    --verified-at YYYY-MM-DD --out benchmark/questions.jsonl --report benchmark/provenance/build_report.json
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_benchmark.py
```

* Re-running `verify` on the current candidates after a refresh shows which evidence quotes or links broke.
  Fix or drop those items; replace them from the reserve (`"select": false` items) or write new ones.
* For a new academic year, update the calendar items (new calendar PDF) and the EBS links (`.../555/<year>`).
* Any change to `questions.jsonl` changes its hash: old runs stay valid for the old set, but are not
  comparable with runs on the new one (the leaderboard warns when sets differ). Re-run the models you
  want to compare.

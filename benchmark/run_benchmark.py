#!/usr/bin/env python3
"""Run the CengBot benchmark on a model and grade it.

Answers are generated through the production path (exactly what
``src/ai_model_worker.py`` does, without RabbitMQ or the database):
``detect_language`` -> ``prompt_guard.guard_input`` -> blocked? refusal :
``generate_response_with_info(..., raise_errors=True, language=...)`` ->
``prompt_guard.guard_output``.

Examples (GPU jobs must hold the shared GPU lock):

  flock <scratch>/gpu.lock .venv/bin/python benchmark/run_benchmark.py                # models/active-model
  flock <scratch>/gpu.lock .venv/bin/python benchmark/run_benchmark.py --model-path models/my-new-model --label qwen2.5-3b-v2
  .venv/bin/python benchmark/run_benchmark.py --regrade benchmark/results/<run>      # re-grade, no GPU
  .venv/bin/python benchmark/run_benchmark.py --no-llm ...                           # deterministic scoring only

Outputs ``benchmark/results/<label-or-model>_<timestamp>/``: answers.jsonl,
judgments.jsonl, scores.jsonl, summary.json, report.md, and refreshes
``benchmark/LEADERBOARD.md`` from every run's summary.json.
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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
SRC = ROOT / "src"
for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import grader  # noqa: E402

DEFAULT_QUESTIONS = _HERE / "questions.jsonl"
DEFAULT_RESULTS = _HERE / "results"
DEFAULT_LEADERBOARD = _HERE / "LEADERBOARD.md"

# The worker's apology when the guard gives no refusal text (src/ai_model_worker.FAILED_ANSWER).
FAILED_ANSWER = {
    "tr": "Üzgünüm, sorunu şu anda yanıtlayamadım. Lütfen biraz sonra tekrar dene.",
    "en": "Sorry, I couldn't answer your question right now. Please try again later.",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def git_info(root: Path = ROOT) -> Dict[str, Any]:
    """Commit + dirty flag (read-only git commands)."""
    info: Dict[str, Any] = {"commit": None, "dirty": None, "modified": []}
    try:
        info["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                        timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                capture_output=True, text=True, timeout=20, check=True).stdout
        modified = [line[3:] for line in status.splitlines() if line.strip()]
        info["dirty"] = bool(modified)
        info["modified"] = modified[:50]
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def relativize(obj: Any) -> Any:
    """Replace absolute paths inside the project with project-relative ones (no local paths in results)."""
    root = str(ROOT)
    if isinstance(obj, dict):
        return {k: relativize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativize(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(root + os.sep):
        return obj[len(root) + 1:]
    if isinstance(obj, str) and root in obj:
        return obj.replace(root + os.sep, "").replace(root, ".")
    return obj


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_items(items: List[Dict[str, Any]], ids: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        by_id = {it["id"]: it for it in items}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"unknown item ids: {', '.join(missing)}")
        items = [by_id[i] for i in wanted]
    if limit is not None:
        items = items[: max(0, limit)]
    return items


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-") or "model"


# ---------------------------------------------------------------------------
# generation (production path)
# ---------------------------------------------------------------------------

def load_production_model(model_path: str):
    """Import the real handler with MODEL_PATH set and offline mode on; returns (model, guard)."""
    os.environ["MODEL_PATH"] = model_path  # relative paths are resolved against the project root
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import prompt_guard  # noqa: WPS433 (runtime import on purpose)
    from llama_model_handler import model_instance

    if not model_instance.load_model():
        raise SystemExit(f"model could not be loaded: {model_instance.last_load_error}")
    return model_instance, prompt_guard


def _worker_language(model, text: str) -> str:
    detect = getattr(model, "detect_language", None)
    if callable(detect):
        try:
            return "tr" if detect(text) == "tr" else "en"
        except Exception:
            pass
    return "tr"


def _seed_everything(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch missing (tests with a fake model)
        pass


def generate_answer(model, guard, question: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Mirror RabbitMQWorker._handle: language -> guard_input -> generate -> guard_output."""
    started = time.monotonic()
    language = _worker_language(model, question)
    result = guard.guard_input(question, language)
    record: Dict[str, Any] = {"detected_language": language, "status": "ok", "info": {}}
    if not result.allowed:
        record.update(answer=result.refusal or FAILED_ANSWER[language], status="blocked",
                      guard_reason=result.reason)
    else:
        if seed is not None:
            _seed_everything(seed)
        try:
            generate = getattr(model, "generate_response_with_info", None)
            if callable(generate):
                answer, info = generate(result.text, raise_errors=True, language=language)
            else:
                answer, info = model.generate_response(result.text), {}
            answer = guard.guard_output(answer, language)
            record.update(answer=answer, info=info or {})
            is_fallback = getattr(guard, "is_fallback_answer", None)
            if callable(is_fallback) and is_fallback(answer):
                record["status"] = "output_fallback"
        except Exception as exc:  # the worker would retry; the benchmark records the failure
            record.update(answer="", status="error", error=f"{type(exc).__name__}: {str(exc)[:300]}")
    record["latency_s"] = round(time.monotonic() - started, 3)
    return record


def generation_settings(model) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    keys = ("temperature", "top_p", "top_k", "max_new_tokens", "repetition_penalty", "max_sentences",
            "max_input_tokens", "prompt_style", "precision", "merge_lora")
    return {k: getattr(cfg, k) for k in keys if cfg is not None and hasattr(cfg, k)}


def model_display_name(model_info: Dict[str, Any], fallback: str) -> str:
    base = model_info.get("base_model")
    version = model_info.get("version") or fallback
    if model_info.get("model_type") == "lora" and base:
        return f"{base} + {version}"
    return str(version or base or fallback)


def run_generation(items: List[Dict[str, Any]], model, guard, run_dir: Path, seed: Optional[int],
                   log: Callable[[str], None] = print) -> List[Dict[str, Any]]:
    answers = []
    path = run_dir / "answers.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for n, item in enumerate(items):
            rec = generate_answer(model, guard, item["question"], None if seed is None else seed + n)
            info = rec.pop("info", {}) or {}
            row = {
                "id": item["id"], "lang": item["lang"], "question": item["question"], "answer": rec["answer"],
                "status": rec["status"], "detected_language": rec["detected_language"],
                "latency_s": rec["latency_s"],
                "input_tokens": info.get("input_tokens"), "output_tokens": info.get("output_tokens"),
                "generation_time_s": info.get("generation_time_s"), "stopped_by": info.get("stopped_by"),
            }
            for key in ("guard_reason", "error"):
                if rec.get(key):
                    row[key] = relativize(rec[key])
            answers.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            log(f"[{n + 1}/{len(items)}] {item['id']} {rec['status']} {rec['latency_s']:.1f}s")
    return answers


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------

def make_judge(run_dir: Path, use_llm: bool, judge_model: str, effort: Optional[str],
               client_factory: Optional[Callable[[], Any]] = None, log: Callable[[str], None] = print):
    """Return (judge or None, judge_meta)."""
    meta: Dict[str, Any] = {"model": judge_model, "rubric_version": grader.RUBRIC_VERSION, "effort": effort,
                            "deterministic_version": grader.DETERMINISTIC_VERSION, "enabled": False,
                            "reason": None}
    if not use_llm:
        meta["reason"] = "disabled with --no-llm"
        return None, meta
    import llm_api

    client = client_factory() if client_factory else llm_api.make_client()
    if client is None:
        meta["reason"] = "no ANTHROPIC_API_KEY / CLAUDE_API_KEY configured"
        log(f"LLM judge disabled: {meta['reason']}; using deterministic scoring")
        return None, meta
    if client_factory is None:  # real client: cheap credential check before grading
        ok, why = llm_api.check_access(client, judge_model)
        if not ok:
            meta["reason"] = why
            log(f"LLM judge disabled: {why}; using deterministic scoring")
            return None, meta
    cache = grader.JudgmentCache(run_dir / "judgments.jsonl")
    meta["enabled"] = True
    return grader.Judge(client, cache, model=judge_model, effort=effort), meta


def grade(items: List[Dict[str, Any]], answers: List[Dict[str, Any]], judge) -> List[Dict[str, Any]]:
    by_id = {a["id"]: a for a in answers}
    results = []
    for item in items:
        ans = by_id.get(item["id"])
        if ans is None:
            continue
        text = ans.get("answer") or ""
        det = grader.deterministic_checks(item, text)
        judgment = judge.judge(item, text) if judge is not None else None
        res = grader.score_item(item, text, judgment, det)
        res["status"] = ans.get("status")
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# report + leaderboard
# ---------------------------------------------------------------------------

def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "-"
    if pct:
        return f"{100 * value:.1f}%"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def write_report(run_dir: Path, summary: Dict[str, Any], items: List[Dict[str, Any]],
                 answers: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> None:
    s = summary["scores"]
    mode = s.get("scoring_mode")
    lines = [
        f"# Benchmark run `{summary['run']}`",
        "",
        f"- Model: **{summary['model_name']}** (`{summary['model'].get('resolved_path') or summary['model'].get('configured_path')}`)",
        f"- Questions: {s['n']} of {summary['questions']['n_total']} (`{summary['questions']['file']}`, sha256 {summary['questions']['sha256'][:12]})",
        f"- Scoring: **{mode}**" + (f" (judge `{summary['judge']['model']}`, rubric {summary['judge']['rubric_version']})"
                                     if summary["judge"]["enabled"] else
                                     f" (LLM judge off: {summary['judge'].get('reason')})"),
        f"- Git: `{(summary['git'].get('commit') or '?')[:12]}`" + (" (working tree dirty)" if summary["git"].get("dirty") else ""),
        f"- Generated: {summary.get('created_at')}, graded: {summary.get('graded_at')}",
        "",
        "## Scores",
        "",
        "| metric | value |",
        "|---|---|",
        f"| overall (0-100) | {_fmt(s.get('overall'))} |",
    ]
    for lang, value in s.get("by_lang", {}).items():
        lines.append(f"| {lang.upper()} | {_fmt(value)} |")
    if s.get("mean_correctness") is not None:
        lines.append(f"| mean correctness (0-4) | {s['mean_correctness']:.2f} |")
    lines += [
        f"| key-fact coverage | {_fmt(s.get('key_fact_coverage'), True)} |",
        f"| link accuracy (correct link present) | {_fmt(s.get('link_accuracy'), True)} |",
        f"| wrong official link rate | {_fmt(s.get('wrong_link_rate'), True)} |",
        f"| hallucinated URL rate | {_fmt(s.get('hallucinated_url_rate'), True)} |",
        f"| hallucination rate | {_fmt(s.get('hallucination_rate'), True)} |",
        f"| language match | {_fmt(s.get('language_ok_rate'), True)} |",
        f"| length sanity | {_fmt(s.get('length_ok_rate'), True)} |",
        "",
        "## By category",
        "",
        "| category | score |",
        "|---|---|",
    ]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in s.get("by_category", {}).items()]
    lines += ["", "| difficulty | score |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in s.get("by_difficulty", {}).items()]
    gen = summary.get("generation") or {}
    if gen:
        lines += ["", f"Generation: mean latency {_fmt(gen.get('mean_latency_s'))} s, "
                      f"blocked {gen.get('blocked', 0)}, errors {gen.get('errors', 0)}, seed {gen.get('seed')}."]
    by_item = {it["id"]: it for it in items}
    by_ans = {a["id"]: a for a in answers}
    worst = sorted(results, key=lambda r: (r["score"], r["id"]))[:10]
    lines += ["", "## 10 worst items", ""]
    for r in worst:
        item, ans = by_item[r["id"]], by_ans.get(r["id"], {})
        flags = []
        if r.get("hallucination"):
            flags.append("hallucination")
        if not r.get("link_found"):
            flags.append("no correct link")
        if r.get("wrong_official_urls"):
            flags.append("wrong link")
        if not r.get("language_ok"):
            flags.append("language")
        lines += [
            f"### {r['id']} ({r['category']}, {r['difficulty']}) score {r['score']:.1f}"
            + (f", correctness {r['correctness']}/4" if r.get("correctness") is not None else ""),
            "",
            f"**Q:** {item['question']}",
            "",
            f"**Answer:** {(ans.get('answer') or '').strip() or '(empty)'}",
            "",
            f"**Reference:** {item['reference_answer']}",
            "",
        ]
        if flags:
            lines += [f"Flags: {', '.join(flags)}", ""]
        if r.get("rationale"):
            lines += [f"Judge: {r['rationale']}", ""]
    (run_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def update_leaderboard(results_dir: Path, path: Path, n_total: Optional[int] = None) -> None:
    """Rebuild LEADERBOARD.md from every run's summary.json (full runs only)."""
    rows, partial = [], 0
    for summary_path in sorted(Path(results_dir).glob("*/summary.json")):
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scores = s.get("scores", {})
        total = s.get("questions", {}).get("n_total")
        if not scores.get("n") or (total and scores["n"] < total):
            partial += 1
            continue
        rows.append(s)

    def table(subset: List[Dict[str, Any]]) -> List[str]:
        out = ["| date | model | overall | TR | EN | link accuracy | hallucination rate | judge | run dir |",
               "|---|---|---|---|---|---|---|---|---|"]
        for s in sorted(subset, key=lambda x: -(x["scores"].get("overall") or 0)):
            sc = s["scores"]
            det_version = s["judge"].get("deterministic_version", "?")
            judge = (f"{s['judge']['model']} ({s['judge']['rubric_version']})" if sc.get("scoring_mode") == "llm"
                     else ("mixed" if sc.get("scoring_mode") == "mixed"
                           else f"none (deterministic {det_version})"))
            dirty = " *" if s.get("git", {}).get("dirty") else ""
            out.append(
                f"| {(s.get('created_at') or '')[:10]} | {s.get('model_name')} | {_fmt(sc.get('overall'))} | "
                f"{_fmt(sc.get('by_lang', {}).get('tr'))} | {_fmt(sc.get('by_lang', {}).get('en'))} | "
                f"{_fmt(sc.get('link_accuracy'), True)} | {_fmt(sc.get('hallucination_rate'), True)} | {judge} | "
                f"[{s['run']}](results/{s['run']}/report.md){dirty} |")
        return out

    llm = [s for s in rows if s["scores"].get("scoring_mode") == "llm"]
    det = [s for s in rows if s["scores"].get("scoring_mode") != "llm"]
    lines = ["# CengBot benchmark leaderboard", "",
             "Generated by `benchmark/run_benchmark.py` from `benchmark/results/*/summary.json`. "
             "Scores are 0-100 (see `benchmark/README.md`). Only compare runs scored the same way: "
             "LLM-judged scores and deterministic (`--no-llm`) scores use different formulas, and LLM-judged "
             "runs are only comparable with the same judge model and rubric version.", "",
             "## LLM-judged", ""]
    lines += table(llm) if llm else ["_No LLM-judged runs yet._"]
    lines += ["", "## Deterministic only (`--no-llm` or no API key)", ""]
    lines += table(det) if det else ["_No deterministic runs yet._"]
    lines += ["", "`*` = the run was made from a working tree with uncommitted changes."]
    question_sets = sorted({(s.get("questions") or {}).get("sha256", "?")[:12] for s in rows})
    if len(question_sets) > 1:
        lines += ["Warning: these runs used different versions of questions.jsonl (sha256 "
                  + ", ".join(question_sets) + "); only compare runs with the same question set "
                  "(see each run's summary.json)."]
    if partial:
        lines += [f"{partial} partial run(s) (`--limit` / `--ids`) are not listed."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default="models/active-model",
                   help="model directory (sets MODEL_PATH for the handler); default models/active-model")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    p.add_argument("--limit", type=int, help="only the first N items (partial run, not on the leaderboard)")
    p.add_argument("--ids", help="comma-separated item ids (partial run)")
    p.add_argument("--no-llm", action="store_true", help="deterministic scoring only (no API calls)")
    p.add_argument("--regrade", metavar="RUN_DIR", help="re-grade an existing run's answers (no generation)")
    p.add_argument("--label", help="run name prefix (default: model directory name)")
    p.add_argument("--seed", type=int, default=1234, help="per-item seed = seed + index (sampling is on)")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    p.add_argument("--leaderboard", default=str(DEFAULT_LEADERBOARD))
    p.add_argument("--judge-model", default=grader.JUDGE_MODEL)
    p.add_argument("--judge-effort", default="medium", choices=["low", "medium", "high"])
    return p


def main(argv: Optional[Sequence[str]] = None, *, model=None, guard=None,
         client_factory: Optional[Callable[[], Any]] = None, log: Callable[[str], None] = print) -> int:
    """CLI entry point. ``model``/``guard``/``client_factory`` let tests inject fakes."""
    args = build_parser().parse_args(argv)
    questions_path = Path(args.questions)
    all_items = read_jsonl(questions_path)
    items = select_items(all_items, args.ids, args.limit)
    results_dir = Path(args.results_dir)
    q_meta = {"file": _rel(questions_path), "sha256": file_sha256(questions_path), "n_total": len(all_items)}

    if args.regrade:
        run_dir = Path(args.regrade)
        old = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) \
            if (run_dir / "summary.json").exists() else {}
        answers = read_jsonl(run_dir / "answers.jsonl")
        answered = {a["id"] for a in answers}
        items = [it for it in items if it["id"] in answered]
        model_info = old.get("model", {})
        model_name = old.get("model_name") or run_dir.name
        created_at = old.get("created_at")
        generation = old.get("generation", {})
        git = old.get("git") or git_info()
        old_sha = (old.get("questions") or {}).get("answers_generated_with") or \
            (old.get("questions") or {}).get("sha256")
        if old_sha and old_sha != q_meta["sha256"]:
            q_meta["answers_generated_with"] = old_sha
            log("warning: questions.jsonl changed since this run was generated")
    else:
        if model is None:
            model, guard = load_production_model(args.model_path)
        elif guard is None:
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            import prompt_guard as guard  # noqa: WPS433
        try:
            model_info = model.get_model_info()
        except Exception as exc:  # pragma: no cover - defensive
            model_info = {"error": str(exc)}
        model_info = relativize(model_info)
        model_info["generation_settings"] = generation_settings(model)
        model_name = model_display_name(model_info, Path(args.model_path).name)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        default_label = model_info.get("version") or Path(args.model_path).name
        run_dir = results_dir / f"{_slug(args.label or default_label)}_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=False)
        created_at = _now()
        git = git_info()
        log(f"run dir: {_rel(run_dir)}; model: {model_name}; items: {len(items)}")
        t0 = time.monotonic()
        answers = run_generation(items, model, guard, run_dir, args.seed, log=log)
        latencies = [a["latency_s"] for a in answers if a["status"] != "blocked"]
        generation = {
            "seed": args.seed, "total_time_s": round(time.monotonic() - t0, 1),
            "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "blocked": sum(1 for a in answers if a["status"] == "blocked"),
            "errors": sum(1 for a in answers if a["status"] == "error"),
            "output_fallbacks": sum(1 for a in answers if a["status"] == "output_fallback"),
            "label": args.label,
        }

    judge, judge_meta = make_judge(run_dir, not args.no_llm, args.judge_model, args.judge_effort,
                                   client_factory=client_factory, log=log)
    results = grade(items, answers, judge)
    write_jsonl(run_dir / "scores.jsonl", results)
    if judge is not None:
        import llm_api

        judge_meta["stats"] = dict(judge.stats)
        judge_meta["usage"] = dict(judge.usage)
        judge_meta["estimated_cost_usd"] = round(llm_api.estimate_cost(judge.model, dict(judge.usage)), 4)
        if judge.disabled_reason:
            judge_meta["reason"] = judge.disabled_reason
    scores = grader.aggregate(results)
    summary = {
        "run": run_dir.name,
        "created_at": created_at,
        "graded_at": _now(),
        "model_name": model_name,
        "model": model_info,
        "git": git,
        "questions": q_meta,
        "judge": judge_meta,
        "generation": generation,
        "scores": scores,
    }
    write_json(run_dir / "summary.json", summary)
    write_report(run_dir, summary, items, answers, results)
    update_leaderboard(results_dir, Path(args.leaderboard))
    by_lang = scores.get("by_lang", {})
    log(f"overall {scores.get('overall')} (TR {by_lang.get('tr')}, EN {by_lang.get('en')}), "
        f"link accuracy {scores.get('link_accuracy')}, hallucination rate {scores.get('hallucination_rate')}, "
        f"scoring {scores.get('scoring_mode')} -> {_rel(run_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuum.evals.adapters.dummy import DummyAdapter
from continuum.evals.adapters.openai_compat import OpenAICompatAdapter
from continuum.evals.graders.rag import grade_rag_case
from continuum.evals.models import EvalCaseResult, EvalSuite, RESULT_SCHEMA_VERSION, RUN_SCHEMA_VERSION
from continuum.evals.reporting.markdown import build_run_markdown_report, write_text
from continuum.evals.suites.loader import suite_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_run_id() -> str:
    return f"evals-run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _format_sources(case_sources: list[Any]) -> str:
    if not case_sources:
        return "(no sources provided)"
    return "\n".join(f"[{source.id}] {source.text}" for source in case_sources)


def _build_messages(template: str, *, question: str, sources: str) -> list[dict[str, str]]:
    content = template.format(question=question, sources=sources)
    return [{"role": "user", "content": content}]


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _adapter_for_model_spec(model_spec: str):
    if model_spec == "dummy":
        return DummyAdapter()
    if model_spec.startswith("http://") or model_spec.startswith("https://"):
        return OpenAICompatAdapter()
    raise RuntimeError(f"Unsupported model spec: {model_spec!r} (use 'dummy' or an http(s) endpoint URL)")


def evals_runs_dir(cwd: Path | None = None) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    return base / ".hydra" / "evals" / "runs"


def run_suite(
    *,
    suite_name: str,
    suite: EvalSuite,
    model_spec: str,
    cwd: Path,
    verbose: bool = False,
    progress_cb=None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    run_id = _build_run_id()
    run_dir = evals_runs_dir(cwd) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    spath = suite_path(suite_name, cwd)
    if spath.exists():
        (run_dir / "suite_snapshot.yaml").write_text(spath.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        (run_dir / "suite_snapshot.yaml").write_text(json.dumps(suite.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    adapter = _adapter_for_model_spec(model_spec)
    results_rows: list[dict[str, Any]] = []

    for idx, case in enumerate(sorted(suite.cases, key=lambda c: c.case_id), start=1):
        sources_text = _format_sources(case.sources)
        messages = _build_messages(suite.model_prompt_template, question=case.question, sources=sources_text)
        p_hash = _prompt_hash(messages)
        output_text = ""
        latency_ms = 0
        error: str | None = None
        started = time.perf_counter()
        try:
            response = adapter.complete(model_spec=model_spec, messages=messages, case_context=case.metadata)
            output_text = response.text
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)

        grades = grade_rag_case(case, output_text if error is None else "")
        passed = bool(grades.get("pass")) and error is None

        result = EvalCaseResult(
            schema_version=RESULT_SCHEMA_VERSION,
            case_id=case.case_id,
            prompt_hash=p_hash,
            output_text=output_text,
            latency_ms=latency_ms,
            error=error,
            grades=grades,
            passed=passed,
        )
        row = result.to_dict()
        results_rows.append(row)
        if progress_cb is not None:
            progress_cb(idx, len(suite.cases), row)
        elif verbose:
            print(f"[evals] {case.case_id}: {'PASS' if passed else 'FAIL'}")

    total = len(results_rows)
    pass_count = sum(1 for row in results_rows if row.get("pass"))
    citation_present = sum(1 for row in results_rows if row.get("grades", {}).get("citation_present"))
    grounded = sum(1 for row in results_rows if row.get("grades", {}).get("grounded"))
    idk = sum(1 for row in results_rows if row.get("grades", {}).get("idk_detected"))
    forbidden_hits = sum(len(row.get("grades", {}).get("forbidden_hits", [])) for row in results_rows)

    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": _utc_now(),
        "suite": suite_name,
        "model_spec": model_spec,
        "run_path": str(run_dir),
        "total_cases": total,
        "pass_count": pass_count,
        "pass_rate": (pass_count / total) if total else 0.0,
        "citation_present_rate": (citation_present / total) if total else 0.0,
        "grounded_rate": (grounded / total) if total else 0.0,
        "idk_rate": (idk / total) if total else 0.0,
        "forbidden_hits": forbidden_hits,
    }

    _write_jsonl(run_dir / "results.jsonl", results_rows)
    _write_json(run_dir / "summary.json", summary)
    write_text(run_dir / "report.md", build_run_markdown_report(summary, results_rows))
    return summary, results_rows, run_dir


def load_run(run_ref: str, cwd: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    base = cwd if cwd is not None else Path.cwd()
    candidate = Path(run_ref)
    run_dir = candidate if candidate.exists() else (evals_runs_dir(base) / run_ref)
    if not run_dir.exists():
        raise FileNotFoundError(f"Eval run not found: {run_ref}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with (run_dir / "results.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return summary, rows, run_dir


def compare_runs(left_ref: str, right_ref: str, cwd: Path | None = None) -> dict[str, Any]:
    left_summary, left_rows, left_dir = load_run(left_ref, cwd)
    right_summary, right_rows, right_dir = load_run(right_ref, cwd)
    left_by_id = {row["case_id"]: bool(row.get("pass")) for row in left_rows}
    right_by_id = {row["case_id"]: bool(row.get("pass")) for row in right_rows}
    all_ids = sorted(set(left_by_id) | set(right_by_id))
    pass_to_fail = [cid for cid in all_ids if left_by_id.get(cid) is True and right_by_id.get(cid) is False]
    fail_to_pass = [cid for cid in all_ids if left_by_id.get(cid) is False and right_by_id.get(cid) is True]
    return {
        "left_summary": left_summary | {"run_path": str(left_dir)},
        "right_summary": right_summary | {"run_path": str(right_dir)},
        "pass_rate_delta": float(right_summary.get("pass_rate", 0.0)) - float(left_summary.get("pass_rate", 0.0)),
        "citation_present_rate_delta": float(right_summary.get("citation_present_rate", 0.0)) - float(left_summary.get("citation_present_rate", 0.0)),
        "grounded_rate_delta": float(right_summary.get("grounded_rate", 0.0)) - float(left_summary.get("grounded_rate", 0.0)),
        "idk_rate_delta": float(right_summary.get("idk_rate", 0.0)) - float(left_summary.get("idk_rate", 0.0)),
        "forbidden_hits_delta": int(right_summary.get("forbidden_hits", 0)) - int(left_summary.get("forbidden_hits", 0)),
        "pass_to_fail": pass_to_fail,
        "fail_to_pass": fail_to_pass,
    }


__all__ = ["run_suite", "compare_runs", "load_run", "evals_runs_dir"]

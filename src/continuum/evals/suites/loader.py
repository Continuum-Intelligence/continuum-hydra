from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from continuum.evals.models import EvalSuite, SUITE_SCHEMA_VERSION


class SuiteError(Exception):
    pass


def _load_yaml_or_json_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise SuiteError("Suite file must contain a mapping/object at top level")
        return data
    except ModuleNotFoundError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - error path
            raise SuiteError(
                "PyYAML not installed and suite file is not JSON-compatible YAML. "
                "Install pyyaml or use JSON syntax in .yaml."
            ) from exc
        if not isinstance(data, dict):
            raise SuiteError("Suite file must contain an object")
        return data


def suites_dir(cwd: Path | None = None) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    return base / "hydra-evals" / "suites"


def suite_path(suite_name: str, cwd: Path | None = None) -> Path:
    return suites_dir(cwd) / f"{suite_name}.yaml"


def load_suite(name: str, cwd: Path | None = None) -> EvalSuite:
    path = suite_path(name, cwd)
    if not path.exists():
        raise SuiteError(f"Suite not found: {path}")
    raw = _load_yaml_or_json_text(path.read_text(encoding="utf-8"))
    suite = EvalSuite.from_dict(raw)
    if suite.schema_version != SUITE_SCHEMA_VERSION:
        raise SuiteError(f"Unsupported suite schema_version: {suite.schema_version!r}")
    if not suite.cases:
        raise SuiteError("Suite has no cases")
    return suite


def init_core_suite(cwd: Path | None = None) -> Path:
    base = cwd if cwd is not None else Path.cwd()
    suite_dir = suites_dir(base)
    suite_dir.mkdir(parents=True, exist_ok=True)
    path = suite_dir / "core.yaml"
    if path.exists():
        return path

    # JSON syntax is valid YAML and avoids a hard dependency on PyYAML for round-trip tests.
    payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "name": "core",
        "model_prompt_template": (
            "Answer the question using only the provided sources. "
            "Cite sources using [S#]. If information is missing, say \"I don't know\".\n\n"
            "Sources:\n{sources}\n\nQuestion: {question}\nAnswer:"
        ),
        "cases": [
            {
                "case_id": "core-pass-1",
                "question": "What year was Project Hydra founded?",
                "sources": [{"id": "S1", "text": "Project Hydra was founded in 2024 by a small research team."}],
                "must_cite": True,
                "must_say_idk_if_insufficient": False,
                "must_not_say": ["hallucinated"],
                "must_be_grounded": True,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = ["SuiteError", "load_suite", "init_core_suite", "suite_path", "suites_dir"]

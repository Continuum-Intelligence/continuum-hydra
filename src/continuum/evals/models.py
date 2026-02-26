from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUITE_SCHEMA_VERSION = "evals.suite.v1"
RUN_SCHEMA_VERSION = "evals.run.v1"
RESULT_SCHEMA_VERSION = "evals.result.v1"


@dataclass(slots=True)
class EvalSource:
    id: str
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSource":
        return cls(id=str(data["id"]), text=str(data["text"]))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text}


@dataclass(slots=True)
class EvalCase:
    case_id: str
    question: str
    sources: list[EvalSource] = field(default_factory=list)
    must_cite: bool = True
    must_say_idk_if_insufficient: bool = False
    must_not_say: list[str] = field(default_factory=list)
    must_be_grounded: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        sources = [EvalSource.from_dict(item) for item in data.get("sources", [])]
        return cls(
            case_id=str(data.get("case_id") or data.get("id")),
            question=str(data["question"]),
            sources=sources,
            must_cite=bool(data.get("must_cite", True)),
            must_say_idk_if_insufficient=bool(data.get("must_say_idk_if_insufficient", False)),
            must_not_say=[str(item) for item in data.get("must_not_say", [])],
            must_be_grounded=bool(data.get("must_be_grounded", True)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "sources": [s.to_dict() for s in self.sources],
            "must_cite": self.must_cite,
            "must_say_idk_if_insufficient": self.must_say_idk_if_insufficient,
            "must_not_say": list(self.must_not_say),
            "must_be_grounded": self.must_be_grounded,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class EvalSuite:
    schema_version: str
    name: str
    model_prompt_template: str
    cases: list[EvalCase]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalSuite":
        return cls(
            schema_version=str(data.get("schema_version", "")),
            name=str(data.get("name", "unnamed")),
            model_prompt_template=str(data["model_prompt_template"]),
            cases=[EvalCase.from_dict(item) for item in data.get("cases", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "model_prompt_template": self.model_prompt_template,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(slots=True)
class EvalCaseResult:
    schema_version: str
    case_id: str
    prompt_hash: str
    output_text: str
    latency_ms: int
    error: str | None
    grades: dict[str, Any]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "prompt_hash": self.prompt_hash,
            "output_text": self.output_text,
            "latency_ms": int(self.latency_ms),
            "error": self.error,
            "grades": dict(self.grades),
            "pass": bool(self.passed),
        }


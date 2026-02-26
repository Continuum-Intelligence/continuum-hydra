from __future__ import annotations

from continuum.evals.adapters.base import ModelResponse


class DummyAdapter:
    def complete(self, *, model_spec: str, messages: list[dict[str, str]], case_context: dict | None = None) -> ModelResponse:
        case_context = case_context or {}
        if "dummy_error" in case_context:
            raise RuntimeError(str(case_context["dummy_error"]))
        text = str(case_context.get("dummy_output") or "I don't know. [S1]")
        return ModelResponse(text=text, raw={"adapter": "dummy"})


__all__ = ["DummyAdapter"]

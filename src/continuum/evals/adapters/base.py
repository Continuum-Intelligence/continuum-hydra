from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ModelResponse:
    text: str
    raw: dict[str, Any] | None = None


class ModelAdapter(Protocol):
    def complete(self, *, model_spec: str, messages: list[dict[str, str]], case_context: dict[str, Any] | None = None) -> ModelResponse:
        ...


from __future__ import annotations

import json
import urllib.error
import urllib.request

from continuum.evals.adapters.base import ModelResponse


class OpenAICompatAdapter:
    def __init__(self, timeout_seconds: int = 60) -> None:
        self.timeout_seconds = timeout_seconds

    def complete(self, *, model_spec: str, messages: list[dict[str, str]], case_context: dict | None = None) -> ModelResponse:
        endpoint = model_spec if model_spec.rstrip("/").endswith("/v1/chat/completions") else (model_spec.rstrip("/") + "/v1/chat/completions")
        payload = {"model": "continuum-evals", "messages": messages}
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"endpoint request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"endpoint request failed: {type(exc).__name__}: {exc}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("endpoint response missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            text = "".join(parts).strip()
        else:
            text = str(content or "").strip()
        if not text:
            raise RuntimeError("endpoint response missing content text")
        return ModelResponse(text=text, raw=data)


__all__ = ["OpenAICompatAdapter"]

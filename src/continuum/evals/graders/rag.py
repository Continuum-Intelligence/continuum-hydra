from __future__ import annotations

import re

from continuum.evals.models import EvalCase

_CITE_RE = re.compile(r"\[(S[0-9A-Za-z_-]+)\]")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def _normalize(text: str) -> str:
    return text.lower()


def grade_rag_case(case: EvalCase, output_text: str) -> dict[str, object]:
    output = output_text or ""
    source_ids = {source.id for source in case.sources}
    source_text = "\n".join(source.text for source in case.sources)

    cited = set(_CITE_RE.findall(output))
    citation_present = bool(cited & source_ids)
    citation_ok = (not case.must_cite) or citation_present

    insufficient = len(case.sources) == 0
    idk_ok = True
    idk_detected = False
    if case.must_say_idk_if_insufficient and insufficient:
        lowered = _normalize(output)
        idk_detected = ("i don't know" in lowered) or ("insufficient" in lowered)
        idk_ok = idk_detected
    elif insufficient:
        lowered = _normalize(output)
        idk_detected = ("i don't know" in lowered) or ("insufficient" in lowered)

    forbidden_hits = [phrase for phrase in case.must_not_say if phrase.lower() in _normalize(output)]
    forbidden_ok = len(forbidden_hits) == 0

    grounded_ok = True
    ungrounded_numbers: list[str] = []
    if case.must_be_grounded:
        source_blob = source_text.lower()
        output_without_citations = _CITE_RE.sub("", output)
        for token in _NUM_RE.findall(output_without_citations):
            if token.lower() not in source_blob:
                ungrounded_numbers.append(token)
        grounded_ok = len(ungrounded_numbers) == 0

    passed = citation_ok and idk_ok and forbidden_ok and grounded_ok
    return {
        "citation_present": citation_present,
        "citation_ok": citation_ok,
        "grounded": grounded_ok,
        "idk_ok": idk_ok,
        "idk_detected": idk_detected,
        "forbidden_hits": forbidden_hits,
        "forbidden_ok": forbidden_ok,
        "ungrounded_numbers": ungrounded_numbers,
        "pass": passed,
    }


__all__ = ["grade_rag_case"]

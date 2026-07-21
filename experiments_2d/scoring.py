"""Answer and response-policy scoring matched to the pinned baseline protocol."""

from __future__ import annotations

import ast
import json
import re
import warnings
from typing import Any


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def extract_boxed(text: Any) -> str:
    source = clean(text)
    marker = "\\boxed{"
    start = source.find(marker)
    if start < 0:
        return ""
    index = start + len(marker)
    depth = 1
    output: list[str] = []
    while index < len(source):
        character = source[index]
        if character == "{":
            depth += 1
            output.append(character)
        elif character == "}":
            depth -= 1
            if depth == 0:
                break
            output.append(character)
        else:
            output.append(character)
        index += 1
    return "".join(output).strip() if output else source


def _normalize_structured(value: Any) -> Any:
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_normalize_structured(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_structured(item) for key, item in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"[-+]?\d*\.\d+", stripped):
            return float(stripped)
        return stripped
    return value


def _parse_structured(text: Any) -> Any | None:
    source = clean(text)
    if not source:
        return None
    try:
        return _normalize_structured(ast.literal_eval(source))
    except Exception:
        return None


def _normalize_scalar(text: Any) -> str:
    value = re.sub(r"\s+", " ", clean(text)).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    value = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", value)
    value = value.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", value).strip().lower()


def compare_values(prediction: Any, gold: Any) -> bool:
    prediction_structured = _parse_structured(prediction)
    gold_structured = _parse_structured(gold)
    if prediction_structured is not None and gold_structured is not None:
        return prediction_structured == gold_structured
    normalized_gold = _normalize_scalar(gold)
    return bool(normalized_gold) and _normalize_scalar(prediction) == normalized_gold


def score_final_response(raw_text: str, gold: Any) -> tuple[str, bool]:
    boxed = extract_boxed(raw_text)
    return boxed, bool(boxed) and compare_values(boxed, gold)


def has_tool_call(text: str) -> bool:
    source = (text or "").strip().replace("```json", "").replace("```", "").strip()

    def parse(block: str) -> bool:
        try:
            data = json.loads(block)
        except Exception:
            try:
                data = json.loads(block.replace("'", '"'))
            except Exception:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SyntaxWarning)
                        data = ast.literal_eval(block)
                    json.dumps(data)
                except Exception:
                    return False
        return isinstance(data, dict) and "name" in data

    if "<tool_call>" in source:
        body = source.split("<tool_call>", 1)[1]
        body = body.split("</tool_call>", 1)[0].strip()
        if parse(body):
            return True
    if parse(source):
        return True
    return any(parse(match.group(0)) for match in re.finditer(r"\{[\s\S]*?\}", source))


def has_nontrivial_reasoning_before_box(text: str) -> bool:
    source = text or ""
    index = source.find("\\boxed{")
    prefix = source[:index] if index >= 0 else source
    prefix = re.sub(r"\s+", " ", prefix).strip()
    return len(prefix) >= 12 and re.search(r"[A-Za-z]", prefix) is not None

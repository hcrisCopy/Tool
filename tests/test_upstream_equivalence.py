import pytest

from experiments_2d.scoring import compare_values, extract_boxed, has_tool_call
from experiments_2d.upstream import load_upstream_runtime


@pytest.mark.parametrize(
    "text",
    [
        r"\boxed{42}",
        r"prefix \boxed{\text{A {nested} value}} suffix",
        r"answer: \boxed{unfinished",
        r"\boxed{",
        "no box",
    ],
)
def test_box_parser_matches_pinned_upstream(text: str) -> None:
    upstream_utils, _ = load_upstream_runtime()
    assert extract_boxed(text) == upstream_utils.extract_boxed(text)


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"calculate","arguments":{}}</tool_call>',
        "<tool_call>malformed output</tool_call>",
        "reasoning then {'name': 'lookup', 'arguments': {}}",
        '{"other":"value"}',
        "plain answer",
    ],
)
def test_tool_parser_decision_matches_pinned_upstream(text: str) -> None:
    _, upstream_model = load_upstream_runtime()
    assert has_tool_call(text) == (
        upstream_model._parse_tool_call_from_text(text) is not None
    )


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [("42", "42"), ("[1, 2]", "[1,2]"), (r"\text{yes}", "yes"), ("a", "b")],
)
def test_answer_comparison_matches_pinned_upstream(prediction: str, gold: str) -> None:
    upstream_utils, _ = load_upstream_runtime()
    assert compare_values(prediction, gold) == upstream_utils.compare_values(
        prediction, gold
    )

from experiments_2d.scoring import (
    compare_values,
    extract_boxed,
    has_nontrivial_reasoning_before_box,
    has_tool_call,
)


def test_extract_boxed_supports_nested_braces() -> None:
    assert extract_boxed(r"answer: \boxed{\text{A {nested} value}}") == r"\text{A {nested} value}"


def test_unclosed_box_matches_pinned_upstream_behavior() -> None:
    assert extract_boxed(r"answer: \boxed{unfinished") == "unfinished"


def test_compare_values_matches_structured_numeric_values() -> None:
    assert compare_values("['1', 2.0]", "[1, '2.0']")
    assert not compare_values("[1, 3]", "[1, 2]")


def test_policy_detectors() -> None:
    assert has_tool_call('<tool_call>{"name":"calculator","arguments":{}}</tool_call>')
    assert not has_tool_call("<tool_call>malformed output</tool_call>")
    assert has_nontrivial_reasoning_before_box("I calculate the answer first. \\boxed{4}")
    assert not has_nontrivial_reasoning_before_box(r"\boxed{4}")

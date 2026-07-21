"""Reproducible no-tool generation and model-specific necessity labels."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from .config import ExperimentConfig
from .constants import ENV_TO_CATEGORY
from .prompting import initial_messages, render_prompt, tools_for_variant
from .scoring import (
    extract_boxed,
    has_nontrivial_reasoning_before_box,
    has_tool_call,
    score_final_response,
)


NO_TOOL_REJECTION = (
    "Tool use is not available. Solve the problem directly without tools and "
    "provide your final answer in \\boxed{...}."
)
REASONING_REJECTION = (
    "Final answer rejected: reasoning is not allowed in no_reasoning mode. "
    "Retry with final answer in \\boxed{...} only."
)
CONTINUE_NO_REASONING = (
    "Continue. You must do exactly one of these next (no reasoning text):\n"
    "1) Provide one valid tool call.\n"
    "2) Provide final answer in \\boxed{...}."
)


@dataclass
class NoToolState:
    task: dict[str, Any]
    messages: list[dict[str, str]]
    tools: list[dict[str, Any]]
    prompt_hash: str = ""
    rounds: int = 0
    done: bool = False
    final_response: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)


def _sampling_params(config: ExperimentConfig, seed: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        max_tokens=config.generation.max_new_tokens,
        seed=seed,
    )


def _load_engine(config: ExperimentConfig, seed: int) -> tuple[Any, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        config.paths.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    engine = LLM(
        model=str(config.paths.model),
        tokenizer=str(config.paths.model),
        tensor_parallel_size=config.generation.tensor_parallel_size,
        max_model_len=config.generation.max_model_len,
        dtype=config.model.torch_dtype,
        gpu_memory_utilization=config.generation.gpu_memory_utilization,
        seed=seed,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    return engine, tokenizer


def _initial_state(task: dict[str, Any]) -> NoToolState:
    return NoToolState(
        task=task,
        messages=initial_messages(task, no_tool=True),
        tools=tools_for_variant(task, "P_env"),
    )


def _advance_state(state: NoToolState, raw_text: str) -> None:
    state.rounds += 1
    state.messages.append({"role": "assistant", "content": raw_text})
    action: str
    if has_tool_call(raw_text):
        state.messages.append({"role": "user", "content": NO_TOOL_REJECTION})
        action = "tool_rejected"
    elif extract_boxed(raw_text):
        if has_nontrivial_reasoning_before_box(raw_text):
            state.messages.append({"role": "user", "content": REASONING_REJECTION})
            action = "reasoning_rejected"
        else:
            state.final_response = raw_text
            state.done = True
            action = "accepted_final"
    else:
        state.messages.append({"role": "user", "content": CONTINUE_NO_REASONING})
        action = "continue"
    state.trace.append(
        {
            "round": state.rounds,
            "raw_text": raw_text,
            "action": action,
        }
    )


def generate_no_tool_labels(
    tasks: list[dict[str, Any]], config: ExperimentConfig, seed: int
) -> list[dict[str, Any]]:
    """Run the audited hard-no-tool state machine without executing any tool."""

    if not tasks:
        raise ValueError("No tasks supplied for no-tool labeling")
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task ids in no-tool labeling input")

    engine, tokenizer = _load_engine(config, seed)
    params = _sampling_params(config, seed)
    states = [_initial_state(task) for task in tasks]
    progress = tqdm(total=len(states), desc=f"no-tool seed={seed}", unit="task")

    for _round in range(1, config.generation.max_rounds + 1):
        active = [state for state in states if not state.done]
        if not active:
            break
        prompts: list[str] = []
        for state in active:
            prompt = render_prompt(tokenizer, state.messages, state.tools)
            if not state.prompt_hash:
                state.prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            prompts.append(prompt)
        generated = engine.generate(prompts, params, use_tqdm=False)
        if len(generated) != len(active):
            raise RuntimeError(
                f"vLLM returned {len(generated)} outputs for {len(active)} prompts"
            )
        for state, request_output in zip(active, generated, strict=True):
            if len(request_output.outputs) != 1:
                raise RuntimeError(
                    f"Task {state.task['id']} returned {len(request_output.outputs)} sequences"
                )
            was_done = state.done
            _advance_state(state, request_output.outputs[0].text)
            if state.done and not was_done:
                progress.update(1)
        progress.set_postfix(active=sum(not state.done for state in states))
    progress.close()

    results: list[dict[str, Any]] = []
    for state in states:
        gold = state.task["expected"]["answer"]
        boxed, correct = score_final_response(state.final_response, gold)
        env_name = state.task["environments"][0]["name"]
        results.append(
            {
                "id": state.task["id"],
                "difficulty": state.task["difficulty"],
                "env": env_name,
                "category": ENV_TO_CATEGORY[env_name],
                "tool_type": ENV_TO_CATEGORY[env_name],
                "seed": seed,
                "prompt_variant": "P_env",
                "prompt_hash": state.prompt_hash,
                "rounds": state.rounds,
                "completed": state.done,
                "final_response": state.final_response,
                "boxed_answer": boxed,
                "gold_answer": gold,
                "no_tool_correct": int(correct),
                "tool_necessary": int(not correct),
                "trace": state.trace,
            }
        )
    if [result["id"] for result in results] != ids:
        raise AssertionError("No-tool output order changed")
    return results


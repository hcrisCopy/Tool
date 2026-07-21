"""Pinned When2Tool hard-no-tool labels with execution disabled by construction."""

from __future__ import annotations

import gc
import hashlib
import os
import random
from typing import Any

import numpy as np
import torch

from .config import ExperimentConfig
from .constants import ENV_TO_CATEGORY
from .upstream import EXPECTED_COMMIT, load_upstream_runtime


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_pinned_agent(config: ExperimentConfig, seed: int) -> Any:
    """Construct the exact upstream AgentModel, adding only an explicit LLM seed.

    The pinned vLLM backend leaves its seed unspecified.  We explicitly set a
    preregistered seed so all three runs are reproducible; no parser, prompt,
    state-machine, or sampling parameter is changed.
    """

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    multiprocessing = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if multiprocessing not in (None, "0"):
        raise ValueError(
            "VLLM_ENABLE_V1_MULTIPROCESSING must be 0 for seeded reproducibility"
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    _, upstream_model = load_upstream_runtime()
    import vllm

    original_llm = vllm.LLM

    def seeded_llm(*args: Any, **kwargs: Any) -> Any:
        if "seed" in kwargs:
            raise ValueError("Pinned upstream unexpectedly supplied its own vLLM seed")
        kwargs["seed"] = seed
        return original_llm(*args, **kwargs)

    # VLLMAgentBackend imports LLM inside __init__, so this scoped replacement
    # preserves the pinned implementation and only exposes the repeat seed.
    vllm.LLM = seeded_llm
    try:
        agent = upstream_model.AgentModel(
            model_path=str(config.paths.model),
            backend="vllm",
            max_new_tokens=config.generation.max_new_tokens,
            tensor_parallel_size=config.generation.tensor_parallel_size,
            max_model_len=config.generation.max_model_len,
            vllm_dtype=config.model.torch_dtype,
            enable_thinking=False,
        )
    finally:
        vllm.LLM = original_llm
    agent.max_model_len = config.generation.max_model_len
    generation_config = agent.engine.generation_config
    expected = {
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "top_k": config.generation.top_k,
        "repetition_penalty": config.generation.repetition_penalty,
        "do_sample": config.generation.do_sample,
    }
    actual = {
        key: getattr(generation_config, key, None) for key in expected
    }
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value is None or float(actual_value) != float(expected_value):
            raise ValueError(
                f"Model generation_config.{key}={actual_value} != frozen {expected_value}"
            )
    if config.generation.gpu_memory_utilization != 0.90:
        raise ValueError(
            "Pinned When2Tool vLLM backend uses its default gpu_memory_utilization=0.90"
        )
    return agent


def _validate_initial_prompts(tasks: list[dict[str, Any]], agent: Any) -> None:
    """Fail before generation if any official prompt cannot be rendered exactly."""

    upstream_utils, _ = load_upstream_runtime()
    for task in tasks:
        state = upstream_utils.init_state(
            task,
            agent.system_prompt,
            record_mode="lite",
            prompt_mode="hard_no_tool",
            require_reasoning=False,
            tool_format="xml",
            tokenizer=agent.tokenizer,
        )
        prompt = agent.render_prompt(state["messages"], state["tools"])
        token_ids = agent.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"Task {task['id']} rendered an empty hard-no-tool prompt")
        if len(token_ids) > agent.max_model_len - 128:
            raise ValueError(
                f"Task {task['id']} hard-no-tool prompt has {len(token_ids)} tokens; "
                f"limit is {agent.max_model_len - 128}"
            )


def _normalize_output(
    task: dict[str, Any], output: dict[str, Any], seed: int
) -> dict[str, Any]:
    upstream_utils, _ = load_upstream_runtime()
    if output.get("id") != task.get("id"):
        raise ValueError(
            f"Upstream output id {output.get('id')} != input id {task.get('id')}"
        )
    trace = output.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"Task {task['id']} has no upstream inference trace")
    prompt_text = trace[0].get("prompt_text")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError(f"Task {task['id']} has no first-round prompt text")
    raw, boxed, cleaned, correct = upstream_utils.item_final_eval(output)
    env_name = task["environments"][0]["name"]
    if env_name not in ENV_TO_CATEGORY:
        raise KeyError(f"Task {task['id']} has unmapped environment {env_name}")
    if int(output.get("tool_calls", -1)) != 0:
        raise AssertionError(
            f"hard_no_tool task {task['id']} executed {output.get('tool_calls')} tools"
        )
    return {
        "id": task["id"],
        "difficulty": task["difficulty"],
        "env": env_name,
        "category": ENV_TO_CATEGORY[env_name],
        "tool_type": ENV_TO_CATEGORY[env_name],
        "seed": seed,
        "prompt_variant": "P_env",
        "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "rounds": int(output["rounds"]),
        "completed": bool(output.get("final_response"))
        and output.get("final_response") != "[CONTEXT_LENGTH_EXCEEDED]",
        "final_response": raw,
        "boxed_answer": boxed,
        "cleaned_answer": cleaned,
        "gold_answer": task["expected"]["answer"],
        "no_tool_correct": int(correct),
        "tool_necessary": int(not correct),
        "tool_calls": 0,
        "generation_tokens": int(output.get("generation_tokens", 0)),
        "prefill_tokens": int(output.get("prefill_tokens", 0)),
        "reasoning_mode": output.get("reasoning_mode"),
        "upstream_commit": EXPECTED_COMMIT,
        "enable_thinking": False,
        "trace": trace,
    }


def generate_no_tool_labels(
    tasks: list[dict[str, Any]], config: ExperimentConfig, seed: int
) -> list[dict[str, Any]]:
    """Run pinned hard-no-tool evaluation while forbidding every tool execution."""

    if not tasks:
        raise ValueError("No tasks supplied for no-tool labeling")
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task ids in no-tool labeling input")
    if seed not in config.generation.seeds:
        raise ValueError(f"Label seed {seed} is not configured: {config.generation.seeds}")

    _seed_everything(seed)
    upstream_utils, _ = load_upstream_runtime()
    agent = _build_pinned_agent(config, seed)
    try:
        _validate_initial_prompts(tasks, agent)

        def forbidden_route(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("Tool execution reached during hard_no_tool labeling")

        original_route = upstream_utils.route_tool_call
        upstream_utils.route_tool_call = forbidden_route
        try:
            upstream_outputs = upstream_utils.evaluate_batched(
                tasks,
                agent,
                max_rounds=config.generation.max_rounds,
                record_mode="lite",
                prompt_mode="hard_no_tool",
                require_reasoning=False,
                tool_format="xml",
            )
        finally:
            upstream_utils.route_tool_call = original_route
        if len(upstream_outputs) != len(tasks):
            raise RuntimeError(
                f"Pinned evaluator returned {len(upstream_outputs)} outputs for "
                f"{len(tasks)} tasks"
            )
        results = [
            _normalize_output(task, output, seed)
            for task, output in zip(tasks, upstream_outputs, strict=True)
        ]
        if [row["id"] for row in results] != ids:
            raise AssertionError("No-tool output order changed")
        return results
    finally:
        del agent
        gc.collect()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

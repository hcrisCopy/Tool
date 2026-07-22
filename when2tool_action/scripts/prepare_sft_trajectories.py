"""Create the filtered full-menu action-SFT JSONL and audit manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from when2tool_action.config import load_config, require_inputs
from when2tool_action.data import load_task_json
from when2tool_action.io_utils import sha256_file
from when2tool_action.provenance import validate_runtime_provenance
from when2tool_action.sft import (
    build_sft_trajectories,
    read_json_artifact,
    write_sft_artifacts,
)
from when2tool_action.upstream import full_menu_sha256, load_runtime


def _source(path: Path) -> dict[str, object]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "file": source.name,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def _load_stage_runtime_provenance(config: object, path: Path) -> dict[str, object]:
    return validate_runtime_provenance(config, path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Derive action SFT data from 900 full-menu train tasks, seed-0 "
            "hard-no-tool answers, and seed-0 full-record scoped trajectories."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--runtime-provenance",
        type=Path,
        required=True,
        help="Stage-07 runtime provenance receipt; the older global receipt is forbidden.",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--scoped-data",
        type=Path,
        required=True,
        help="The exact scoped train JSON used to generate --scoped-trajectories.",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--no-tool-generations", type=Path, required=True)
    parser.add_argument("--scoped-trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Default: replace the output .jsonl suffix with .manifest.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    config = load_config(args.config)
    require_inputs(config)
    data_path = args.data.resolve()
    scoped_data_path = args.scoped_data.resolve()
    labels_path = args.labels.resolve()
    no_tool_path = args.no_tool_generations.resolve()
    scoped_path = args.scoped_trajectories.resolve()
    output_path = args.output.resolve()
    runtime_provenance_path = args.runtime_provenance.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else output_path.with_suffix(".manifest.json")
    )
    inputs = (data_path, scoped_data_path, labels_path, no_tool_path, scoped_path)
    if len(set(inputs)) != len(inputs):
        raise ValueError("SFT input artifacts must be five distinct files")
    if runtime_provenance_path in inputs:
        raise ValueError(
            "Runtime provenance must be distinct from SFT source artifacts"
        )
    if output_path in (*inputs, runtime_provenance_path) or manifest_path in (
        *inputs,
        runtime_provenance_path,
    ):
        raise ValueError("SFT outputs must not overwrite source artifacts")

    runtime_receipt = _load_stage_runtime_provenance(config, runtime_provenance_path)
    tasks = load_task_json(data_path, expected_scope="full")
    scoped_tasks = load_task_json(scoped_data_path, expected_scope="scoped")
    labels_artifact = read_json_artifact(labels_path)
    no_tool_artifact = read_json_artifact(no_tool_path)
    scoped_artifact = read_json_artifact(scoped_path)
    utils, _model_module, _registry = load_runtime()
    tool_format = utils.detect_tool_format(str(config.paths.model))
    if tool_format != "xml":
        raise ValueError(
            f"Qwen3 SFT construction requires XML tools, got {tool_format}"
        )
    system_prompt = utils.get_system_prompt(tool_format)
    sources = {
        "data": _source(data_path),
        "scoped_data": _source(scoped_data_path),
        "labels": _source(labels_path),
        "no_tool_generations": _source(no_tool_path),
        "scoped_trajectories": _source(scoped_path),
    }
    result = build_sft_trajectories(
        tasks,
        scoped_tasks,
        labels_artifact,
        no_tool_artifact,
        scoped_artifact,
        system_prompt=system_prompt,
        source_provenance=sources,
        expected_model_slug=config.model.slug,
        expected_full_menu_sha256=full_menu_sha256(),
        runtime_provenance={
            "file": Path(runtime_receipt["path"]).name,
            "sha256": runtime_receipt["sha256"],
            "project_git_commit": runtime_receipt["git_commit"],
        },
    )
    manifest = write_sft_artifacts(
        output_path, manifest_path, result, overwrite=args.overwrite
    )
    decisions = manifest["decisions"]
    print(
        f"Published {decisions['retained']} retained / {decisions['total']} source "
        f"tasks to {output_path}; dropped={decisions['dropped']}"
    )
    print(f"Audit manifest: {manifest_path}")


if __name__ == "__main__":
    main()

"""Run P_no_schema necessity and P_all tool-type onset scans."""

from __future__ import annotations

import argparse
import json

import torch

from experiments_2d.config import load_config
from experiments_2d.constants import HIDDEN_PROTOCOL_REVISION
from experiments_2d.io_utils import (
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)
from experiments_2d.onset import curve_rows, run_onset_analysis
from experiments_2d.plotting import plot_necessity_onset, plot_type_onset
from experiments_2d.upstream import (
    EXPECTED_COMMIT,
    all_candidate_menu_sha256,
)


CURVE_FIELDS = [
    "layer",
    "necessity_A",
    "necessity_B",
    "necessity_C",
    "necessity_common",
    "type_A",
    "type_B",
    "type_C",
    "type_overall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments_2d/configs/qwen3_4b_instruct_2507.yaml",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--n-shuffles", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--label-seed",
        type=int,
        default=None,
        help="Generation seed for hard-no-tool labels (default: first configured seed).",
    )
    return parser.parse_args()


def _load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_manifest_fields(path, expected):
    manifest = _load_json(path)
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise ValueError(
                f"{path}: {key}={manifest.get(key)!r} != {expected_value!r}"
            )
    return manifest


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA onset analysis requested but CUDA is unavailable")
    config = load_config(args.config)
    label_seed = config.generation.seeds[0] if args.label_seed is None else args.label_seed
    if label_seed not in config.generation.seeds:
        raise ValueError(
            f"--label-seed {label_seed} is not one of {config.generation.seeds}"
        )
    n_shuffles = (
        config.analysis.n_label_shuffles_primary
        if args.n_shuffles is None
        else args.n_shuffles
    )
    if n_shuffles < 2:
        raise ValueError("--n-shuffles must be at least 2")
    hidden_root = config.run_root / "hidden" / args.mode
    necessity_path = hidden_root / "P_no_schema" / "train_hidden.pt"
    type_path = hidden_root / "P_all" / "train_hidden.pt"
    label_path = (
        config.run_root
        / "labels"
        / args.mode
        / f"seed_{label_seed}"
        / "train_no_tool_outputs.json"
    )
    label_manifest_path = label_path.with_name("train_manifest.json")
    label_manifest = _require_manifest_fields(
        label_manifest_path,
        {
            "split": "train",
            "mode": args.mode,
            "seed": label_seed,
            "upstream_commit": EXPECTED_COMMIT,
            "enable_thinking": False,
            "max_new_tokens": config.generation.max_new_tokens,
            "max_rounds": config.generation.max_rounds,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "repetition_penalty": config.generation.repetition_penalty,
            "do_sample": config.generation.do_sample,
            "vllm_enable_v1_multiprocessing": False,
            "single_gpu_adaptation": True,
        },
    )
    necessity_metadata = _load_json(
        hidden_root / "P_no_schema" / "train_metadata.json"
    )
    type_metadata = _load_json(hidden_root / "P_all" / "train_metadata.json")
    labels = _load_json(label_path)
    if label_manifest.get("n") != len(labels):
        raise ValueError("Label manifest count differs from label rows")
    if any(row.get("seed") != label_seed for row in labels):
        raise ValueError("At least one label row has the wrong generation seed")
    if any(row.get("upstream_commit") != EXPECTED_COMMIT for row in labels):
        raise ValueError("At least one label row is not from the pinned evaluator")
    menu_sha256 = all_candidate_menu_sha256()
    for variant in ("P_no_schema", "P_all"):
        _require_manifest_fields(
            hidden_root / variant / "train_manifest.json",
            {
                "split": "train",
                "mode": args.mode,
                "prompt_variant": variant,
                "dtype": "torch.float32",
                "extraction_batch_size": config.extraction_batch_size,
                "enable_thinking": False,
                "protocol_revision": HIDDEN_PROTOCOL_REVISION,
                "p_all_menu_sha256": menu_sha256,
            },
        )
    necessity_ids = [row["id"] for row in necessity_metadata]
    type_ids = [row["id"] for row in type_metadata]
    label_ids = [row["id"] for row in labels]
    if necessity_ids != type_ids or necessity_ids != label_ids:
        raise ValueError("P_no_schema, P_all, and label task order differ")

    hidden_necessity = torch.load(necessity_path, map_location="cpu", weights_only=True)
    hidden_type = torch.load(type_path, map_location="cpu", weights_only=True)
    expected_shape = (
        len(labels),
        config.model.num_hidden_layers + 1,
        config.model.hidden_size,
    )
    if tuple(hidden_necessity.shape) != expected_shape:
        raise ValueError(
            f"P_no_schema hidden shape {tuple(hidden_necessity.shape)} != {expected_shape}"
        )
    if tuple(hidden_type.shape) != expected_shape:
        raise ValueError(f"P_all hidden shape {tuple(hidden_type.shape)} != {expected_shape}")
    output_dir = (
        config.run_root
        / "onset"
        / args.mode
        / f"label_seed_{label_seed}"
        / f"shuffles_{n_shuffles}"
    )
    expected_outputs = (
        output_dir / "onset_summary.json",
        output_dir / "onset_curves.csv",
        output_dir / "residual_scalers.pt",
        output_dir / "onset_tool_necessity_write_signal.png",
        output_dir / "onset_category_write_signal.png",
    )
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Onset outputs already exist; archive them before rerunning: "
            + ", ".join(str(path) for path in existing)
        )
    result, scalers = run_onset_analysis(
        hidden_necessity,
        hidden_type,
        labels,
        n_shuffles=n_shuffles,
        seed=config.analysis.seed,
        peak_fraction=config.analysis.onset_peak_fraction,
        max_window=config.analysis.max_onset_window,
        device=args.device,
    )
    result["label_seed"] = label_seed
    result["inputs"] = {
        "labels_sha256": sha256_file(label_path),
        "necessity_hidden_sha256": sha256_file(necessity_path),
        "type_hidden_sha256": sha256_file(type_path),
        "hidden_protocol_revision": HIDDEN_PROTOCOL_REVISION,
        "p_all_menu_sha256": menu_sha256,
        "upstream_commit": EXPECTED_COMMIT,
    }
    atomic_write_json(output_dir / "onset_summary.json", result)
    atomic_write_csv(
        output_dir / "onset_curves.csv",
        curve_rows(result),
        fieldnames=CURVE_FIELDS,
    )
    atomic_torch_save(output_dir / "residual_scalers.pt", scalers)
    plot_necessity_onset(
        result["necessity"], output_dir / "onset_tool_necessity_write_signal.png"
    )
    plot_type_onset(
        result["type"], output_dir / "onset_category_write_signal.png"
    )
    print(json.dumps({
        "necessity_onset": result["necessity"]["onset"],
        "type_onset": result["type"]["onset"],
        "clean_set_counts": result["clean_set_counts"],
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()

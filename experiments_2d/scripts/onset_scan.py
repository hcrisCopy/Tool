"""Run P_no_schema necessity and P_all tool-type onset scans."""

from __future__ import annotations

import argparse
import json

import torch

from experiments_2d.config import load_config
from experiments_2d.io_utils import atomic_torch_save, atomic_write_csv, atomic_write_json
from experiments_2d.onset import curve_rows, run_onset_analysis
from experiments_2d.plotting import plot_necessity_onset, plot_type_onset


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
    return parser.parse_args()


def _load_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA onset analysis requested but CUDA is unavailable")
    config = load_config(args.config)
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
        / f"seed_{config.generation.seeds[0]}"
        / "train_no_tool_outputs.json"
    )
    necessity_metadata = _load_json(
        hidden_root / "P_no_schema" / "train_metadata.json"
    )
    type_metadata = _load_json(hidden_root / "P_all" / "train_metadata.json")
    labels = _load_json(label_path)
    necessity_ids = [row["id"] for row in necessity_metadata]
    type_ids = [row["id"] for row in type_metadata]
    label_ids = [row["id"] for row in labels]
    if necessity_ids != type_ids or necessity_ids != label_ids:
        raise ValueError("P_no_schema, P_all, and label task order differ")

    hidden_necessity = torch.load(necessity_path, map_location="cpu", weights_only=True)
    hidden_type = torch.load(type_path, map_location="cpu", weights_only=True)
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
    output_dir = config.run_root / "onset" / args.mode / f"shuffles_{n_shuffles}"
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

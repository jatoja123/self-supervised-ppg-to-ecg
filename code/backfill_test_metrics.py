"""Backfill missing held-out test metrics on the original W&B runs.

Run from the repository root. W&B decides which metrics are missing; local
best_model.pt files provide the weights and the exact training data settings.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval import evaluate_dataset
from train import (ModelConfig, PPGtoECGGenerator, PairedPPGECGDataset,
                   load_training_data, set_seed, split_subjects)


METRICS = {
    "FD_mean": "test/eval/FD_mean",
    "FD_std": "test/eval/FD_std",
    "RMSE_mean": "test/eval/RMSE_mean_normalized",
    "RMSE_std": "test/eval/RMSE_std_normalized",
    "rho_mean": "test/eval/rho_mean",
    "rho_std": "test/eval/rho_std",
    "MAEHR_Eprime_mean": "test/eval/MAEHR_Eprime_mean_bpm",
    "MAEHR_Eprime_std": "test/eval/MAEHR_Eprime_std_bpm",
    "MAEHR_P_mean": "test/eval/MAEHR_P_mean_bpm",
    "MAEHR_P_std": "test/eval/MAEHR_P_std_bpm",
    "valid_HR_segments": "test/eval/valid_HR_segments",
}
PROVENANCE = "test/eval/checkpoint_best_epoch"
CHECK_CONFIG = ("model_name", "sampling_rate", "window_seconds", "overlap",
                "data_loader", "data_fraction", "seed", "in_channels",
                "out_channels", "base_channels", "num_unet_layers",
                "gen_transformer_dim", "gen_transformer_layers",
                "gen_transformer_heads", "gen_transformer_mlp_ratio",
                "max_tokens")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="ppg-to-ecg", help="W&B project")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), help="W&B user/team")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRICS), default=list(METRICS),
                        help="Metrics from code/eval.py to backfill (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="List eligible runs without evaluating or writing")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def local_run_id(directory: Path) -> str | None:
    wandb_dir = directory / "wandb"
    if not wandb_dir.is_dir():
        return None
    ids = {match.group(1) for child in wandb_dir.iterdir()
           if (match := re.fullmatch(r"run-\d{8}_\d{6}-([a-z0-9]+)", child.name))}
    return next(iter(ids)) if len(ids) == 1 else None


def index_checkpoints(results_dir: Path):
    by_id, by_name = {}, {}
    if not results_dir.is_dir():
        raise FileNotFoundError(f"No results directory: {results_dir}")
    for path in results_dir.glob("*/best_model.pt"):
        directory = path.parent
        by_name[directory.name] = path
        run_id = local_run_id(directory)
        if run_id:
            if run_id in by_id:
                raise ValueError(f"Ambiguous local W&B run ID: {run_id}")
            by_id[run_id] = path
    return by_id, by_name


def checkpoint_for_run(run, by_id, by_name):
    if run.id in by_id:
        return by_id[run.id]
    path = by_name.get(run.name)
    # A directory with another run's ID must never be attached by name alone.
    if path and local_run_id(path.parent) is None:
        return path
    return None


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "generator_state_dict" not in checkpoint:
        raise ValueError(f"Not a GAN generator checkpoint: {path}")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError(f"Missing checkpoint configuration: {path}")
    allowed = {field.name for field in fields(ModelConfig)}
    cfg = ModelConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    return checkpoint, cfg


def matches_wandb_config(run, cfg):
    for key in CHECK_CONFIG:
        remote = run.config.get(key, getattr(ModelConfig(), key))
        local = getattr(cfg, key)
        if remote != local:
            return False, f"{key}: W&B={remote!r}, checkpoint={local!r}"
    return True, ""


def evaluate_checkpoint(checkpoint, cfg, device):
    set_seed(cfg.seed)
    ppg, ecg, subject_ids = load_training_data(
        cfg.data_loader, cfg.data_dir, sampling_rate=cfg.sampling_rate,
        window_seconds=cfg.window_seconds, overlap_ratio=cfg.overlap,
        data_fraction=cfg.data_fraction, seed=cfg.seed,
        held_out_subject=cfg.held_out_subject,
    )
    _, _, test_indices = split_subjects(subject_ids, cfg.held_out_subject, cfg.seed)
    if not len(test_indices):
        raise ValueError("Empty held-out test split")
    print(f"  Test: {len(test_indices)} windows; subjects={np.unique(subject_ids[test_indices]).tolist()}")
    loader = DataLoader(PairedPPGECGDataset(ppg[test_indices], ecg[test_indices]),
                        batch_size=cfg.batch_size, shuffle=False)
    model = PPGtoECGGenerator(cfg).to(device)
    model.load_state_dict(checkpoint["generator_state_dict"], strict=True)
    model.eval()
    generated = []
    with torch.inference_mode():
        for batch_ppg, _ in loader:
            generated.append(model(batch_ppg.to(device)).cpu().numpy()[:, 0, :])
    _, summary = evaluate_dataset(ecg[test_indices], np.concatenate(generated),
                                  fs=cfg.sampling_rate, ppg_segments=ppg[test_indices])
    return summary


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    import wandb

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if not entity:
        raise ValueError("Cannot determine W&B entity; pass --entity")
    by_id, by_name = index_checkpoints(args.results_dir)
    print(f"Found {len(by_name)} local checkpoints; scanning {entity}/{args.project}")
    selected = {METRICS[name] for name in args.metrics}
    processed = skipped = failed = 0
    for run in api.runs(f"{entity}/{args.project}"):
        missing = selected.difference(run.summary.keys())
        if not missing:
            continue
        path = checkpoint_for_run(run, by_id, by_name)
        if path is None:
            print(f"SKIP {run.id} ({run.name}): no matching local checkpoint")
            skipped += 1
            continue
        try:
            checkpoint, cfg = load_checkpoint(path)
            matches, reason = matches_wandb_config(run, cfg)
            if not matches:
                raise ValueError(f"W&B/checkpoint config mismatch: {reason}")
            print(f"RUN {run.id} ({run.name}): {len(missing)} missing metrics, {path}")
            if args.dry_run:
                print("  Missing: " + ", ".join(sorted(missing)))
                processed += 1
                continue
            summary = evaluate_checkpoint(checkpoint, cfg, device)
            updates = {}
            for field, key in METRICS.items():
                if key in missing:
                    value = getattr(summary, field)
                    updates[key] = (float(value) if value is not None and math.isfinite(value)
                                    else None)
            if PROVENANCE not in run.summary:
                updates[PROVENANCE] = checkpoint.get("best_epoch")
            # Public API edits the original run summary without resuming its history.
            run.summary.update(updates)
            print(f"  Uploaded {len(updates)} summary fields to {run.url}")
            processed += 1
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            print(f"ERROR {run.id}: {exc}")
            failed += 1
    print(f"Done: processed={processed}, skipped={skipped}, failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

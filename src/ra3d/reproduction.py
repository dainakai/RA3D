"""Frozen-data evaluation and nested refitting of the paper's collision models."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .config import DEVELOPMENT_SEEDS, TERMINAL_SEED, paper_recipe
from .io import read_table, write_table, write_json
from .evaluation import event_curve
from .training import load_paper_training, fit_staged, nested_base_scores, fit_residuals
from .models import load_model
from .cascade import predict_full
from ._core.context import context_frame
from ._core.relation import fit_model as fit_relation


def evaluate_published(root, output):
    root = Path(root)
    output = Path(output)
    summary = {}
    for dimension in ["3d", "2d"]:
        method = "relation_attention_" + dimension
        for split, seeds in [("train", DEVELOPMENT_SEEDS), ("validation", (TERMINAL_SEED,))]:
            predictions = {
                s: read_table(root / "predictions" / dimension / split / f"seed{s}.parquet") for s in seeds
            }
            truths = {
                s: read_table(root / "labels" / split / f"seed{s}" / "collision_events.parquet")
                for s in seeds
            }
            curve, metrics = event_curve(predictions, truths, score_column=method)
            name = f"{dimension}_{'nested_oof' if split == 'train' else 'terminal'}"
            write_table(curve, output / (name + "_pr.parquet"))
            summary[name] = metrics
    write_json(summary, output / "metrics.json")
    return summary


def refit_paper(root, output, *, device="cpu", cpu_threads=4, reuse_scores=False, progress=None):
    """Refit the 3D deployment model and five outer relation/residual folds.

    Upstream candidate tables are the published scene-wise cross-fitted intermediates.
    Reuse of released OOF scores is explicit and recorded; the default recomputes them.
    """
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stages, manifest = load_paper_training(root)
    full = stages["full"]
    truth = stages["truth"]
    model_path = output / "model"
    recipe = paper_recipe()
    recipe["device"] = device
    with threadpool_limits(limits=cpu_threads):
        if (model_path / "model.json").exists():
            bundle = load_model(model_path, device=device)
        else:
            bundle, _ = fit_staged(
                stages,
                model_path,
                device=device,
                cpu_threads=cpu_threads,
                progress=progress,
                reuse_oof=root / "scores" if reuse_scores else None,
                source_manifest=manifest,
                verify_paper=True,
            )
        score_root = root / "scores" if reuse_scores else output / "nested_scores"
        if not reuse_scores:
            nested_base_scores(full, score_root, progress=progress)
        predictions = {}
        for outer in DEVELOPMENT_SEEDS:
            path = output / "predictions" / f"seed{outer}.parquet"
            if path.exists():
                predictions[outer] = read_table(path)
                continue
            train = [s for s in DEVELOPMENT_SEEDS if s != outer]
            context = [
                context_frame(
                    full[s], np.load(score_root / f"nested_outer{outer}_seed{s}.npy", allow_pickle=False)
                )
                for s in train
            ]
            relation, mean, std, audit = fit_relation(context, recipe)
            residual = fit_residuals(full, truth, train, recipe["residual_model"])
            fold = {
                **bundle,
                "relation_model": relation,
                "relation_mean": mean,
                "relation_std": std,
                "residual_models": residual,
                "recipe": recipe,
            }
            base = np.load(model_path / f"oof_{outer}.npy", allow_pickle=False)
            predictions[outer], policy = predict_full(full[outer], fold, scene_id=outer, base_scores=base)
            write_table(predictions[outer], path)
            write_json(
                {
                    "outer_scene": outer,
                    "training_scenes": train,
                    "reused_base_scores": reuse_scores,
                    "relation": audit,
                    "policy": policy,
                },
                path.with_suffix(".json"),
            )
            del context, relation, residual, fold
            gc.collect()
        curve, development = event_curve(predictions, truth)
        write_table(curve, output / "nested_oof_pr.parquet")
        terminal = read_table(root / "features/validation" / f"seed{TERMINAL_SEED}" / "paper92.parquet")
        terminal_truth = read_table(
            root / "labels/validation" / f"seed{TERMINAL_SEED}" / "collision_events.parquet"
        )
        terminal_pred, _ = predict_full(terminal, bundle, scene_id=TERMINAL_SEED)
        write_table(terminal_pred, output / "predictions" / f"seed{TERMINAL_SEED}.parquet")
        curve, terminal_metrics = event_curve({TERMINAL_SEED: terminal_pred}, {TERMINAL_SEED: terminal_truth})
        write_table(curve, output / "terminal_pr.parquet")
    result = {
        "3d_nested_oof": development,
        "3d_terminal": terminal_metrics,
        "reused_base_scores": reuse_scores,
        "upstream_tables": "published_scene_cross_fitted",
        "validation_scene_used_for_fitting": False,
    }
    write_json(result, output / "metrics.json")
    return result

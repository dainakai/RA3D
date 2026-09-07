"""Depth-free counterpart: 13 screening, 49 static, and 41 relation features."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
import torch
import sklearn
from itertools import combinations
from safetensors.torch import save_file, load_file
from threadpoolctl import threadpool_limits

from ._core import two_d_common as common
from ._core import two_d_relation as relation
from ._core.two_d_budget import budget_select
from .config import DEVELOPMENT_SEEDS, TERMINAL_SEED, Units
from .features import generate_candidates, static_features
from .io import read_table, write_table, write_json, sha256
from .models import TRUSTED_SKLEARN_TYPES
from .training import fit_classifier, TrainingCheckpoint
from .tracks import read_tracks, materialize_tracks
from .evaluation import event_curve


def score(models, frame):
    from .io import require_columns

    require_columns(frame, common.TWO_D_FEATURES, name="depth-free features")
    matrix = common.feature_frame(frame, list(common.TWO_D_FEATURES))
    return np.mean([model.predict_proba(matrix)[:, 1] for model in models], axis=0)


def select_union(static, static_model):
    result = static.copy()
    values = static_model.predict_proba(common.feature_frame(result, list(common.TWO_D_FEATURES)))[:, 1]
    structural = common.logit_blend(result.stage1_score.to_numpy(float), values, 0.75)
    rank = common.rank_within_vanished(result, structural)
    mask = np.zeros(len(result), dtype=bool)
    mask[np.argsort(-values, kind="stable")[: int(np.ceil(0.25 * len(result)))]] = True
    keep = mask | (rank < 20)
    result = result.loc[keep].reset_index(drop=True)
    result["structural_score"] = structural[keep]
    result["structural_vanished_rank"] = rank[keep]
    result["static_global_top25"] = mask[keep].astype(np.int8)
    if "identity_training_selected" in result:
        result["relation_positive"] = result.identity_training_selected.astype(np.int8)
    return result


def predict(full, bundle, scene_id=0, base_scores=None):
    data = full.reset_index(drop=True).copy()
    data["relation_positive"] = False
    data["matched_truth_event_id"] = -1
    data["synth_seed"] = scene_id
    base = score(bundle["full_models"], data) if base_scores is None else base_scores
    context = relation.context_frame(data, base)
    values, _ = relation.attention_scores(context, bundle["relation"], bundle["mean"], bundle["std"])
    selected, _ = relation.policy_predictions(data, values, scene_id)
    result = relation.correct_predictions(selected, data, bundle["residuals"])
    result["candidate_frame_raw"] = data.iloc[result._source_index.to_numpy(int)].candidate_frame.to_numpy(
        int
    )
    result["score"] = result.relation_attention_2d
    result["candidate_id"] = [
        f"{scene_id}:{int(a)}:{int(b)}:{int(f)}"
        for a, b, f in zip(result.vanished_track_id, result.survivor_track_id, result.candidate_frame_raw)
    ]
    return result.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def save(bundle, directory, audit):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sio.dump(
        {k: bundle[k] for k in ["stage1", "static", "full_models", "residuals"]}, directory / "trees.skops"
    )
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in bundle["relation"].state_dict().items()},
        directory / "relation.safetensors",
    )
    np.savez(directory / "normalization.npz", mean=bundle["mean"], std=bundle["std"])
    write_json(
        {
            "schema_version": 1,
            "method": "depth_free_2d",
            "sklearn_version": sklearn.__version__,
            "features": list(common.TWO_D_FEATURES),
            "files": {
                f: sha256(directory / f) for f in ["trees.skops", "relation.safetensors", "normalization.npz"]
            },
            "audit": audit,
        },
        directory / "model.json",
    )


def load(directory, device="cpu"):
    directory = Path(directory)
    config = json.loads((directory / "model.json").read_text())
    if (
        config.get("schema_version") != 1
        or config.get("method") != "depth_free_2d"
        or config["features"] != list(common.TWO_D_FEATURES)
    ):
        raise ValueError("Incompatible depth-free model")
    if config.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Depth-free model requires its recorded scikit-learn version")
    for name in ["trees.skops", "relation.safetensors", "normalization.npz"]:
        if sha256(directory / name) != config["files"][name]:
            raise ValueError(f"Model checksum mismatch: {name}")
    unknown = set(sio.get_untrusted_types(file=directory / "trees.skops"))
    if not unknown <= TRUSTED_SKLEARN_TYPES:
        raise ValueError("Unsupported baseline model types")
    bundle = sio.load(directory / "trees.skops", trusted=sorted(unknown))
    attention = relation.RelationAttentionRanker(len(relation.RELATION_FEATURES))
    attention.load_state_dict(load_file(directory / "relation.safetensors"))
    attention.to(device).eval()
    with np.load(directory / "normalization.npz", allow_pickle=False) as norm:
        bundle.update(mean=norm["mean"], std=norm["std"])
    bundle["relation"] = attention
    return bundle


def train(root, output, *, device="cpu", reuse_scores=False, cpu_threads=4, progress=print):
    root = Path(root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.json").exists():
        raise FileExistsError("Depth-free model output already exists; choose a new directory")
    frames = {}
    gate = {}
    static = {}
    truth = {}
    manifest = {}
    for seed in DEVELOPMENT_SEEDS:
        for stage, target in [("gate", gate), ("static", static), ("full", frames)]:
            path = root / "baseline2d/train" / f"seed{seed}" / (stage + ".parquet")
            target[seed] = read_table(path)
            manifest[str(path.relative_to(root))] = sha256(path)
        truth_path = root / "labels/train" / f"seed{seed}" / "collision_events.parquet"
        truth[seed] = read_table(truth_path)
        manifest[str(truth_path.relative_to(root))] = sha256(truth_path)
        if reuse_scores:
            score_path = root / "baseline2d/scores" / f"outer_seed{seed}.npy"
            manifest[str(score_path.relative_to(root))] = sha256(score_path)
    checkpoint = TrainingCheckpoint(
        output / "checkpoints",
        {"method": "depth_free_2d", "data": manifest, "device": device, "reuse_scores": reuse_scores},
    )
    emit = progress or (lambda x: None)
    audit = {"training_scenes": list(DEVELOPMENT_SEEDS), "reused_scores": reuse_scores, "data": manifest}
    torch.set_num_threads(cpu_threads)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
    with threadpool_limits(limits=cpu_threads):
        stage1, _ = checkpoint.tree(
            "stage1", lambda: fit_classifier(gate.values(), list(common.STAGE1_FEATURES)), emit
        )
        static_model, _ = checkpoint.tree(
            "static", lambda: fit_classifier(static.values(), list(common.TWO_D_FEATURES)), emit
        )
        bundle = {"stage1": stage1, "static": static_model}
        contexts = []
        for held in DEVELOPMENT_SEEDS:
            if reuse_scores:
                path = root / "baseline2d/scores" / f"outer_seed{held}.npy"
                base = np.load(path, allow_pickle=False)
                audit.setdefault("score_hashes", {})[path.name] = sha256(path)
            else:
                models = []
                for state in [269723, 269724, 269725]:
                    model, _ = checkpoint.tree(
                        f"outer_{held}_{state}",
                        lambda: fit_classifier(
                            (frames[s] for s in DEVELOPMENT_SEEDS if s != held),
                            list(common.TWO_D_FEATURES),
                            random_state=state,
                        ),
                        emit,
                    )
                    models.append(model)
                base = score(models, frames[held])
                del models
                gc.collect()
            if base.shape != (len(frames[held]),) or not np.isfinite(base).all():
                raise ValueError("Invalid baseline OOF scores")
            np.save(output / f"oof_{held}.npy", base)
            contexts.append(relation.context_frame(frames[held], base))
        attention, mean, std, attention_audit = relation.fit_attention(contexts, device)
        bundle.update(relation=attention, mean=mean, std=std)
        audit["relation"] = attention_audit
        del contexts
        gc.collect()
        bundle["full_models"] = []
        for state in [269723, 269724, 269725]:
            model, _ = checkpoint.tree(
                f"full_{state}",
                lambda: fit_classifier(frames.values(), list(common.TWO_D_FEATURES), random_state=state),
                emit,
            )
            bundle["full_models"].append(model)
        bundle["residuals"], audit["residuals"] = checkpoint.tree(
            "residuals", lambda: relation.fit_residual_models(frames, DEVELOPMENT_SEEDS, truth), emit
        )
        save(bundle, output, audit)
    return bundle, audit


def infer(tracks, model, output, *, scene_id=0, static_budget=None, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.parquet").exists():
        raise FileExistsError("Inference output already exists; choose a new directory")
    table = read_tracks(tracks).copy()
    table["z"] = 0.0
    table["vz"] = 0.0
    bundle = load(model, device)
    wide = common.add_2d_derived_features(generate_candidates(table))
    if wide.empty:
        result = pd.DataFrame(
            columns=[*common.KEY_COLUMNS, "x_um", "y_um", "z_um", "candidate_frame_raw", "score"]
        )
        write_table(result, output / "predictions.parquet")
        return result
    stage1 = bundle["stage1"].predict_proba(common.feature_frame(wide, list(common.STAGE1_FEATURES)))[:, 1]
    if static_budget is None:
        ranked = common.attach_outside_rank(wide, stage1, common.selected_gate_mask(wide))
        static_budget = int(common.shortlist_mask(ranked).sum())
    short, _ = budget_select(wide, stage1, static_budget)
    static = common.add_2d_derived_features(
        static_features(short, table, materialize_tracks(table, output / "tracks"), Units())
    )
    full = select_union(static, bundle["static"])
    write_table(full, output / "features49.parquet")
    result = predict(full, bundle, scene_id)
    write_table(result, output / "predictions.parquet")
    return result


def refit(root, output, *, device="cpu", reuse_scores=False, cpu_threads=4, progress=print):
    """Refit nested 2D relation evaluation on frozen scene-wise upstream tables."""
    root, output = Path(root), Path(output)
    frames = {
        s: read_table(root / "baseline2d/train" / f"seed{s}" / "full.parquet") for s in DEVELOPMENT_SEEDS
    }
    truths = {
        s: read_table(root / "labels/train" / f"seed{s}" / "collision_events.parquet")
        for s in DEVELOPMENT_SEEDS
    }
    paths = [root / "baseline2d/train" / f"seed{s}" / "full.parquet" for s in DEVELOPMENT_SEEDS]
    paths += [root / "labels/train" / f"seed{s}" / "collision_events.parquet" for s in DEVELOPMENT_SEEDS]
    if reuse_scores:
        paths += sorted((root / "baseline2d/scores").glob("*.npy"))
    checkpoint = TrainingCheckpoint(
        output / "checkpoints",
        {
            "method": "nested_2d",
            "device": device,
            "reuse_scores": reuse_scores,
            "inputs": {str(p.relative_to(root)): sha256(p) for p in paths},
        },
    )
    emit = progress or (lambda value: None)
    torch.set_num_threads(cpu_threads)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
    with threadpool_limits(limits=cpu_threads):
        model_path = output / "model"
        if (model_path / "model.json").exists():
            bundle = load(model_path, device)
        else:
            bundle, _ = train(
                root,
                model_path,
                device=device,
                reuse_scores=reuse_scores,
                cpu_threads=cpu_threads,
                progress=progress,
            )
        score_root = root / "baseline2d/scores" if reuse_scores else output / "nested_scores"
        score_root.mkdir(parents=True, exist_ok=True)
        if not reuse_scores:
            for a, b in combinations(DEVELOPMENT_SEEDS, 2):
                fits = []
                for state in [269723, 269724, 269725]:
                    fitted, _ = checkpoint.tree(
                        f"nested_{a}_{b}_{state}",
                        lambda: fit_classifier(
                            (frames[s] for s in DEVELOPMENT_SEEDS if s not in [a, b]),
                            list(common.TWO_D_FEATURES),
                            random_state=state,
                        ),
                        emit,
                    )
                    fits.append(fitted)
                for outer, held in [(a, b), (b, a)]:
                    np.save(score_root / f"nested_outer{outer}_seed{held}.npy", score(fits, frames[held]))
                del fits
                gc.collect()
        predictions = {}
        for outer in DEVELOPMENT_SEEDS:
            destination = output / "predictions" / f"seed{outer}.parquet"
            if destination.exists():
                predictions[outer] = read_table(destination)
                continue
            train_seeds = [s for s in DEVELOPMENT_SEEDS if s != outer]
            contexts = [
                relation.context_frame(
                    frames[s], np.load(score_root / f"nested_outer{outer}_seed{s}.npy", allow_pickle=False)
                )
                for s in train_seeds
            ]
            attention, mean, std, audit = relation.fit_attention(contexts, device)
            residual, residual_audit = relation.fit_residual_models(frames, train_seeds, truths)
            fold = {**bundle, "relation": attention, "mean": mean, "std": std, "residuals": residual}
            base = np.load(model_path / f"oof_{outer}.npy", allow_pickle=False)
            predictions[outer] = predict(frames[outer], fold, outer, base)
            write_table(predictions[outer], destination)
            write_json(
                {
                    "training_scenes": train_seeds,
                    "relation": audit,
                    "residuals": residual_audit,
                    "reused_scores": reuse_scores,
                },
                destination.with_suffix(".json"),
            )
            emit({"stage": "nested_2d", "outer_scene": outer})
            del contexts, attention, fold
            gc.collect()
        curve, development = event_curve(predictions, truths)
        write_table(curve, output / "nested_oof_pr.parquet")
        terminal = read_table(root / "baseline2d/validation" / f"seed{TERMINAL_SEED}" / "full.parquet")
        truth = read_table(root / "labels/validation" / f"seed{TERMINAL_SEED}" / "collision_events.parquet")
        prediction = predict(terminal, bundle, TERMINAL_SEED)
        write_table(prediction, output / "predictions" / f"seed{TERMINAL_SEED}.parquet")
        curve, validation = event_curve({TERMINAL_SEED: prediction}, {TERMINAL_SEED: truth})
        write_table(curve, output / "terminal_pr.parquet")
    result = {
        "2d_nested_oof": development,
        "2d_terminal": validation,
        "reused_scores": reuse_scores,
        "upstream_tables": "published_scene_cross_fitted",
    }
    write_json(result, output / "metrics.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--reuse-scores", action="store_true")
    fit.add_argument("--device", default="cpu")
    nested = sub.add_parser("reproduce")
    nested.add_argument("--data", required=True)
    nested.add_argument("--output", required=True)
    nested.add_argument("--reuse-scores", action="store_true")
    nested.add_argument("--device", default="cpu")
    pred = sub.add_parser("predict")
    pred.add_argument("--features", required=True)
    pred.add_argument("--model", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--scene-id", type=int, default=0)
    pred.add_argument("--device", default="cpu")
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--tracks", required=True)
    infer_parser.add_argument("--model", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.add_argument("--scene-id", type=int, default=0)
    infer_parser.add_argument("--static-budget", type=int)
    infer_parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.command == "train":
        train(
            args.data,
            args.output,
            device=args.device,
            reuse_scores=args.reuse_scores,
            progress=lambda x: print(json.dumps(x), flush=True),
        )
    elif args.command == "reproduce":
        refit(
            args.data,
            args.output,
            device=args.device,
            reuse_scores=args.reuse_scores,
            progress=lambda x: print(json.dumps(x), flush=True),
        )
    elif args.command == "predict":
        write_table(
            predict(read_table(args.features), load(args.model, args.device), args.scene_id), args.output
        )
    else:
        infer(
            args.tracks,
            args.model,
            args.output,
            scene_id=args.scene_id,
            static_budget=args.static_budget,
            device=args.device,
        )


if __name__ == "__main__":
    main()

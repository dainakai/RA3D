"""Scene-separated classifier, relation-attention, and coordinate-correction training."""

from __future__ import annotations

import gc
import hashlib
import json
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import skops.io as sio
from safetensors.torch import save_file, load_file
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .config import (
    HGB_PARAMS,
    STAGE1_FEATURES,
    STATIC_FEATURES,
    ENHANCED_FEATURES,
    DEVELOPMENT_SEEDS,
    TERMINAL_SEED,
    paper_recipe,
)
from .io import read_table, write_json, sha256, require_columns
from .models import save_model, TRUSTED_SKLEARN_TYPES
from .cascade import ensemble_scores
from ._core.features import feature_frame
from ._core.context import context_frame
from ._core.relation import fit_model as fit_relation, RelationAttentionRanker


class TrainingCheckpoint:
    """Atomically save completed stages; reject reuse with different training inputs."""

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        marker = self.directory / "identity.json"
        if marker.exists() and json.loads(marker.read_text())["sha256"] != fingerprint:
            raise ValueError("Training checkpoint inputs or settings differ; choose a new output directory")
        write_json({"sha256": fingerprint, "inputs": identity}, marker)

    def tree(self, name, fit, emit):
        path = self.directory / (name + ".skops")
        if path.exists():
            unknown = set(sio.get_untrusted_types(file=path))
            if not unknown <= TRUSTED_SKLEARN_TYPES:
                raise ValueError(f"Unsupported checkpoint types: {sorted(unknown)}")
            saved = sio.load(path, trusted=sorted(unknown))
            emit({"stage": name, "resumed": True})
            return saved["model"], saved["audit"]
        model, audit = fit()
        temporary = path.with_suffix(".partial")
        sio.dump({"model": model, "audit": audit}, temporary)
        temporary.replace(path)
        emit({"stage": name, **audit})
        return model, audit


def _input_identity(stages, manifest):
    if manifest:
        return manifest
    return {
        f"{stage}/{seed}": hashlib.sha256(
            json.dumps(list(frame.columns)).encode()
            + pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
        ).hexdigest()
        for stage, parts in stages.items()
        for seed, frame in parts.items()
    }


def training_rows(parts, features):
    selected = []
    for part in parts:
        require_columns(
            part, list(features) + ["spatial_positive", "identity_training_selected", "identity_event_factor"]
        )
        keep = part.spatial_positive.eq(0) | part.identity_training_selected.astype(bool)
        data = part.loc[keep, list(features)].copy()
        data["train_label"] = part.loc[keep, "identity_training_selected"].astype(np.int8).to_numpy()
        data["train_positive_factor"] = part.loc[keep, "identity_event_factor"].to_numpy(np.float64)
        selected.append(data)
    table = pd.concat(selected, ignore_index=True)
    if set(table.train_label.unique()) != {0, 1}:
        raise ValueError("Each training fold must contain positive and negative examples")
    return table


def sample_weights(table):
    positive = table.train_label.to_numpy(np.int8) == 1
    factors = table.loc[positive, "train_positive_factor"].to_numpy(np.float64)
    if not np.isfinite(factors).all() or (factors <= 0).any():
        raise ValueError("Positive examples require finite positive event weights")
    weights = np.zeros(len(table), dtype=np.float64)
    weights[positive] = factors / factors.sum()
    weights[~positive] = 1.0 / int((~positive).sum())
    return weights / weights.mean()


def fit_classifier(parts, features, *, random_state=269723, parameters=None):
    table = training_rows(parts, features)
    model = HistGradientBoostingClassifier(**(parameters or HGB_PARAMS), random_state=random_state)
    start = time.monotonic()
    model.fit(
        feature_frame(table, list(features)),
        table.train_label.to_numpy(np.int8),
        sample_weight=sample_weights(table),
    )
    audit = {
        "rows": len(table),
        "positive_rows": int(table.train_label.sum()),
        "features": len(features),
        "random_state": random_state,
        "iterations": model.n_iter_,
        "seconds": time.monotonic() - start,
    }
    return model, audit


def fit_ensemble(frames, seeds, *, parameters=None, progress=None):
    models = []
    audits = []
    for state in (269723, 269724, 269725):
        model, audit = fit_classifier(
            (frames[s] for s in seeds), ENHANCED_FEATURES, random_state=state, parameters=parameters
        )
        models.append(model)
        audits.append(audit)
        if progress:
            progress({"stage": "base_classifier", "train_scenes": list(seeds), **audit})
    return models, audits


def fit_residuals(frames, truths, seeds, recipe):
    tables = []
    weights = []
    targets = {key: [] for key in ("frame_delta", "x_delta", "y_delta", "z_delta")}
    for seed in seeds:
        part = frames[seed]
        selected = part.loc[part.identity_training_selected.astype(bool)].copy()
        truth = truths[seed].reset_index(drop=True)
        indices = selected.assigned_truth_index.to_numpy(np.int64)
        if (indices < 0).any() or (indices >= len(truth)).any():
            raise ValueError(f"Invalid truth row index in scene {seed}")
        targets["frame_delta"].append(
            truth.iloc[indices].frame.to_numpy(float) - selected.candidate_frame.to_numpy(float)
        )
        for axis in "xyz":
            targets[axis + "_delta"].append(
                truth.iloc[indices][axis + "_um"].to_numpy(float) - selected[axis + "_um"].to_numpy(float)
            )
        tables.append(selected)
        weights.append(selected.identity_event_factor.to_numpy(np.float64))
    matrix = feature_frame(pd.concat(tables, ignore_index=True), ENHANCED_FEATURES)
    weight = np.concatenate(weights)
    weight /= weight.mean()
    result = {}
    for offset, (name, parts) in enumerate(targets.items()):
        model = HistGradientBoostingRegressor(
            loss="squared_error", early_stopping=False, random_state=270100 + offset, **recipe
        )
        model.fit(matrix, np.concatenate(parts).astype(np.float64), sample_weight=weight)
        result[name] = model
    return result


def load_paper_training(root: str | Path):
    root = Path(root)
    stages = {name: {} for name in ("stage1", "static", "full", "truth")}
    manifest = {}
    for seed in DEVELOPMENT_SEEDS:
        scene = f"seed{seed}"
        paths = {
            "stage1": root / "features/train" / scene / "stage1_training.parquet",
            "static": root / "features/train" / scene / "static77.parquet",
            "full": root / "features/train" / scene / "paper92.parquet",
            "truth": root / "labels/train" / scene / "collision_events.parquet",
        }
        for stage, path in paths.items():
            stages[stage][seed] = read_table(path)
            manifest[str(path.relative_to(root))] = sha256(path)
    return stages, manifest


def fit_staged(
    stages: dict,
    output: str | Path,
    *,
    device="cpu",
    progress=None,
    reuse_oof: str | Path | None = None,
    cpu_threads=4,
    recipe=None,
    parameters=None,
    source_manifest=None,
    verify_paper=False,
):
    """Fit a complete deployable model from precomputed, scene-wise candidate tables.

    OOF scores are recomputed unless reuse_oof explicitly selects a prior audited score directory.
    Use reproduce() for the nested outer evaluation; fitting a deployment model needs one OOF level.
    """
    output = Path(output)
    if (output / "model.json").exists():
        raise FileExistsError(f"Model directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(sorted(stages["full"]))
    if len(seeds) < 3:
        raise ValueError(
            "At least three independent training scenes are required for relation-attention cross-fitting"
        )
    if any(set(stages[name]) != set(seeds) for name in stages):
        raise ValueError("Training stage scene memberships differ")
    if verify_paper and (seeds != DEVELOPMENT_SEEDS or TERMINAL_SEED in seeds):
        raise ValueError("The paper training split must be exactly seeds 260740--260744")
    recipe = dict(recipe or paper_recipe())
    recipe["device"] = device
    torch.set_num_threads(cpu_threads)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
    emit = progress or (lambda value: None)
    checkpoint = TrainingCheckpoint(
        output / "checkpoints",
        {
            "format": 1,
            "scenes": seeds,
            "recipe": recipe,
            "parameters": parameters or HGB_PARAMS,
            "source_manifest": _input_identity(stages, source_manifest),
            "reused_oof": {str(s): sha256(Path(reuse_oof) / f"outer_seed{s}.npy") for s in seeds}
            if reuse_oof
            else None,
        },
    )
    audit = {
        "training_scenes": list(seeds),
        "candidate_tables_fixed": True,
        "source_manifest": source_manifest or {},
        "base_score_reuse": reuse_oof is not None,
        "stages": {},
    }
    with threadpool_limits(limits=cpu_threads):
        bundle = {"recipe": recipe}
        for key, stage, features in [
            ("stage1_model", "stage1", STAGE1_FEATURES),
            ("static_model", "static", STATIC_FEATURES),
        ]:
            model, fit = checkpoint.tree(
                stage,
                lambda: fit_classifier((stages[stage][s] for s in seeds), features, parameters=parameters),
                emit,
            )
            bundle[key] = model
            audit["stages"][stage] = fit
            write_json(audit, output / "training_audit.json")
        full = stages["full"]
        oof = {}
        for held_out in seeds:
            training = [s for s in seeds if s != held_out]
            if reuse_oof:
                path = Path(reuse_oof) / f"outer_seed{held_out}.npy"
                score = np.load(path, allow_pickle=False)
                if score.shape != (len(full[held_out]),) or not np.isfinite(score).all():
                    raise ValueError(f"OOF score shape or finiteness mismatch for scene {held_out}")
                audit.setdefault("reused_score_hashes", {})[path.name] = sha256(path)
            else:
                models = []
                fit = []
                for state in (269723, 269724, 269725):
                    model, entry = checkpoint.tree(
                        f"outer_{held_out}_{state}",
                        lambda: fit_classifier(
                            (full[s] for s in training),
                            ENHANCED_FEATURES,
                            random_state=state,
                            parameters=parameters,
                        ),
                        emit,
                    )
                    models.append(model)
                    fit.append(entry)
                score = ensemble_scores(models, full[held_out])
                audit["stages"][f"outer_{held_out}"] = {"training_scenes": training, "fits": fit}
                del models
                gc.collect()
            oof[held_out] = score
            np.save(output / f"oof_{held_out}.npy", score)
            write_json(audit, output / "training_audit.json")
        contexts = [context_frame(full[s], oof[s]) for s in seeds]
        relation_marker = checkpoint.directory / "relation.json"
        if relation_marker.exists():
            relation = RelationAttentionRanker(46, recipe["hidden_size"], recipe["attention_heads"])
            relation.load_state_dict(load_file(checkpoint.directory / "relation.safetensors"))
            relation.to(device).eval()
            with np.load(checkpoint.directory / "relation_normalization.npz", allow_pickle=False) as norm:
                mean, std = norm["mean"], norm["std"]
            relation_audit = json.loads(relation_marker.read_text())
        else:
            relation, mean, std, relation_audit = fit_relation(contexts, recipe)
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in relation.state_dict().items()},
                checkpoint.directory / "relation.safetensors",
            )
            np.savez(checkpoint.directory / "relation_normalization.npz", mean=mean, std=std)
            write_json(relation_audit, relation_marker)
        bundle.update(relation_model=relation, relation_mean=mean, relation_std=std)
        audit["stages"]["relation"] = relation_audit
        emit({"stage": "relation", **relation_audit})
        del contexts
        gc.collect()
        bundle["full_models"] = []
        fit = []
        for state in (269723, 269724, 269725):
            model, entry = checkpoint.tree(
                f"full_{state}",
                lambda: fit_classifier(
                    (full[s] for s in seeds), ENHANCED_FEATURES, random_state=state, parameters=parameters
                ),
                emit,
            )
            bundle["full_models"].append(model)
            fit.append(entry)
        audit["stages"]["full"] = fit
        bundle["residual_models"], _ = checkpoint.tree(
            "residuals",
            lambda: (fit_residuals(full, stages["truth"], seeds, recipe["residual_model"]), {}),
            emit,
        )
        emit({"stage": "residuals_complete"})
        save_model(bundle, output, provenance=audit)
        write_json(audit, output / "training_audit.json")
    return bundle, audit


def nested_base_scores(full: dict, output: str | Path, *, parameters=None, progress=None):
    """Exclude both the outer evaluation scene and the inner scored scene."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(sorted(full))
    if len(seeds) < 3:
        raise ValueError("Nested evaluation requires at least three independent scenes")
    for excluded_a, excluded_b in combinations(seeds, 2):
        train = tuple(s for s in seeds if s not in (excluded_a, excluded_b))
        models, audits = fit_ensemble(full, train, parameters=parameters, progress=progress)
        for outer, scored in [(excluded_a, excluded_b), (excluded_b, excluded_a)]:
            np.save(output / f"nested_outer{outer}_seed{scored}.npy", ensemble_scores(models, full[scored]))
        write_json(
            {"excluded_scenes": [excluded_a, excluded_b], "training_scenes": train, "fits": audits},
            output / f"nested_{excluded_a}_{excluded_b}.json",
        )
        del models
        gc.collect()

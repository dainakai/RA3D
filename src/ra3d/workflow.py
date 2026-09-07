"""Portable inference from trajectories and scene-wise training preparation."""

from __future__ import annotations

from pathlib import Path
import gc
import json

import yaml
from threadpoolctl import threadpool_limits

from .acquisition import Acquisition, Reconstructor
from .config import Units, STAGE1_FEATURES, STATIC_FEATURES
from .features import generate_candidates, static_features, image_features
from .cascade import shortlist, score_static, image_mask, scoring_union, predict_full, selected_gate
from ._core.features import feature_frame
from ._core.labels import attach_identity_labels
from .models import load_model
from .io import read_table, write_table, write_json, sha256, require_columns, require_unique, require_integer
from .tracks import read_tracks, materialize_tracks


def infer(
    tracks,
    model,
    output,
    *,
    acquisition=None,
    image_bank=None,
    scene_id=0,
    units=Units(),
    device="cpu",
    cpu_threads=4,
    progress=None,
):
    output = Path(output)
    if (output / "predictions.parquet").exists():
        raise FileExistsError("Inference output already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_model(model, device=device)
    table = read_tracks(tracks)
    acq = Acquisition.load(acquisition) if acquisition else None
    if acq:
        units = acq.units
    track_dir = materialize_tracks(table, output / "tracks")
    with threadpool_limits(limits=cpu_threads):
        wide = generate_candidates(table, units)
        write_table(wide, output / "wide.parquet")
        if wide.empty:
            predictions, audit = predict_full(wide, bundle, scene_id=scene_id)
        else:
            score = bundle["stage1_model"].predict_proba(feature_frame(wide, STAGE1_FEATURES))[:, 1]
            short = shortlist(wide, score, bundle["recipe"]["outside_fraction"])
            static = static_features(short, table, track_dir, units, progress)
            static = score_static(static, bundle["static_model"])
            write_table(static, output / "static77.parquet")
            requested = static.loc[image_mask(static, bundle["recipe"]["image_fraction"])].reset_index(
                drop=True
            )
            if image_bank:
                images = read_table(image_bank)
            elif acq:
                images = image_features(
                    requested,
                    table,
                    Reconstructor(acq, device),
                    progress=progress,
                    checkpoint_path=output / "image_checkpoint.npz",
                )
                write_table(images, output / "image_features.parquet")
            else:
                raise ValueError(
                    "Inference requires --acquisition or a complete --image-bank for selected candidates"
                )
            full = scoring_union(static, images, bundle["recipe"])
            write_table(full, output / "paper92.parquet")
            predictions, audit = predict_full(full, bundle, scene_id=scene_id)
        write_table(predictions, output / "predictions.parquet")
        write_table(predictions, output / "predictions.csv")
    write_json(
        {
            "scene_id": scene_id,
            "units": vars(units),
            "model_sha256": sha256(Path(model) / "model.json"),
            "input_tracks_sha256": sha256(tracks) if Path(tracks).is_file() else None,
            "acquisition": str(Path(acquisition).resolve()) if acquisition else None,
            "rows": len(predictions),
            "audit": audit,
        },
        output / "run.json",
    )
    return predictions


def label_candidates(candidates, truth, mapping):
    truth = truth.copy().reset_index(drop=True)
    require_columns(
        truth,
        ["collision_event_id", "frame", "x_um", "y_um", "z_um", "parent1_id", "parent2_id", "child_id"],
        finite=True,
    )
    require_unique(truth, ["collision_event_id"])
    require_integer(
        truth, ["collision_event_id", "frame", "parent1_id", "parent2_id", "child_id"], name="truth"
    )
    require_integer(mapping, ["frame", "observed_track_id", "source_particle_id"], name="identity mapping")
    require_unique(mapping, ["frame", "observed_track_id"], name="identity mapping")
    if "sampling_component" not in truth:
        truth["sampling_component"] = "custom"
    return attach_identity_labels(candidates.reset_index(drop=True), truth, mapping)


def load_project(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema_version") != 1 or not isinstance(cfg.get("scenes"), list):
        raise ValueError("Training project requires schema_version: 1 and a scenes list")
    scenes = {}
    for item in cfg["scenes"]:
        if any(not item.get(key) for key in ["tracks", "truth", "mapping"]):
            raise ValueError(
                "Each scene requires tracks, truth, and mapping paths; an empty identity mapping with headers is allowed"
            )
        if isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int):
            raise ValueError("Each scene ID must be an integer")
        scene = int(item["id"])
        if scene in scenes:
            raise ValueError(f"Duplicate scene ID: {scene}")
        if item.get("split") not in {"train", "validation"}:
            raise ValueError("Each scene split must be train or validation")
        if item.get("exhaustive_truth") is not True:
            raise ValueError(
                "Complete event truth is required to label unmatched candidates negative. Set exhaustive_truth: true only after exhaustive annotation; sparse review labels are insufficient."
            )
        entry = dict(item)
        for key in ["tracks", "acquisition", "truth", "mapping", "image_bank"]:
            if entry.get(key):
                entry[key] = path.parent / entry[key]
        if not entry.get("acquisition") and not entry.get("image_bank"):
            raise ValueError("Each training scene needs an acquisition or a precomputed image bank")
        scenes[scene] = entry
    if sum(s["split"] == "train" for s in scenes.values()) < 3:
        raise ValueError("At least three independent training scenes are required")
    return scenes


def prepare_training(project, output, *, device="cpu", cpu_threads=4, progress=None, parameters=None):
    """Build upstream stages with leave-one-scene-out selection on training scenes."""
    from .training import fit_classifier

    scenes = load_project(project)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prepared.json").exists():
        raise FileExistsError("Prepared training project already exists; choose a new output directory")
    training = tuple(s for s in sorted(scenes) if scenes[s]["split"] == "train")
    wide = {}
    gate = {}
    statics = {}
    truths = {}
    maps = {}
    units = {}
    directories = {}
    audit = {
        "training_scenes": training,
        "validation_scenes": [s for s in scenes if s not in training],
        "upstream_selection": "leave_one_training_scene_out",
        "scenes": {},
    }
    for scene, spec in sorted(scenes.items()):
        root = output / spec["split"] / f"scene{scene}"
        directories[scene] = root
        tracks = read_tracks(spec["tracks"])
        write_table(tracks, root / "tracks.parquet")
        materialize_tracks(tracks, root / "tracks")
        units[scene] = (
            Acquisition.load(spec["acquisition"]).units
            if spec.get("acquisition")
            else Units(**spec.get("units", {}))
        )
        truths[scene] = read_table(spec["truth"])
        maps[scene] = read_table(spec["mapping"])
        write_table(truths[scene], root / "truth.parquet")
        pool = generate_candidates(tracks, units[scene])
        if pool.empty:
            raise ValueError(f"Scene {scene} has no collision candidates")
        wide[scene] = pool
        labeled, label_audit, _ = label_candidates(pool.loc[selected_gate(pool)], truths[scene], maps[scene])
        labeled["synth_seed"] = scene
        gate[scene] = labeled
        write_table(labeled, root / "stage1.parquet")
        audit["scenes"][scene] = {
            "split": spec["split"],
            "wide_candidates": len(pool),
            "gate_labeling": label_audit,
        }
    with threadpool_limits(limits=cpu_threads):
        for scene in sorted(scenes):
            fit_scenes = [s for s in training if s != scene]
            model, fit = fit_classifier((gate[s] for s in fit_scenes), STAGE1_FEATURES, parameters=parameters)
            score = model.predict_proba(feature_frame(wide[scene], STAGE1_FEATURES))[:, 1]
            short = shortlist(wide[scene], score)
            static = static_features(
                short,
                read_tracks(directories[scene] / "tracks.parquet"),
                directories[scene] / "tracks",
                units[scene],
                progress,
            )
            static, label_audit, _ = label_candidates(static, truths[scene], maps[scene])
            static["synth_seed"] = scene
            statics[scene] = static
            write_table(static, directories[scene] / "static.parquet")
            audit["scenes"][scene]["stage1_training_scenes"] = fit_scenes
            audit["scenes"][scene]["static_labeling"] = label_audit
            if progress:
                progress({"stage": "prepare_static", "scene": scene, "rows": len(static), "fit": fit})
        del wide
        gc.collect()
        for scene, spec in sorted(scenes.items()):
            fit_scenes = [s for s in training if s != scene]
            model, fit = fit_classifier(
                (statics[s] for s in fit_scenes), STATIC_FEATURES, parameters=parameters
            )
            scored = score_static(statics[scene], model)
            write_table(scored, directories[scene] / "static_scored.parquet")
            requested = scored.loc[image_mask(scored)].reset_index(drop=True)
            if spec.get("image_bank"):
                image = read_table(spec["image_bank"])
            else:
                image = image_features(
                    requested,
                    read_tracks(directories[scene] / "tracks.parquet"),
                    Reconstructor(Acquisition.load(spec["acquisition"]), device),
                    progress=progress,
                    checkpoint_path=directories[scene] / "image_checkpoint.npz",
                )
            full = scoring_union(scored, image)
            full, label_audit, _ = label_candidates(full, truths[scene], maps[scene])
            full["synth_seed"] = scene
            write_table(full, directories[scene] / "full.parquet")
            audit["scenes"][scene]["static_training_scenes"] = fit_scenes
            audit["scenes"][scene]["full_labeling"] = label_audit
            if progress:
                progress({"stage": "prepare_full", "scene": scene, "rows": len(full), "fit": fit})
    write_json(audit, output / "prepared.json")
    return audit


def load_prepared(root):
    root = Path(root)
    metadata = json.loads((root / "prepared.json").read_text())
    stages = {name: {} for name in ["stage1", "static", "full", "truth"]}
    manifest = {}
    for scene in metadata["training_scenes"]:
        for stage in stages:
            path = root / "train" / f"scene{scene}" / (stage + ".parquet")
            stages[stage][scene] = read_table(path)
            manifest[str(path.relative_to(root))] = sha256(path)
    return stages, manifest

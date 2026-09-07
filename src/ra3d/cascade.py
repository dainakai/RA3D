"""Frozen candidate selection, image acquisition policy, and collision scoring."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .config import (
    KEY_COLUMNS,
    GEOMETRY_COLUMNS,
    IMAGE_FEATURES,
    STATIC_FEATURES,
    ENHANCED_FEATURES,
    paper_recipe,
)
from .features import validate_candidates
from .io import require_columns
from ._core.features import add_derived_features, feature_frame
from ._core.ranking import logit_blend
from ._core.ranks import attach_ranks
from ._core.context import context_frame
from ._core.relation import relation_scores
from ._core.policy import temporal_mode_predictions
from ._core.residual import correct_predictions


def selected_gate(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame.post_survival_frames.ge(1)
        & frame.time_offset_frame.between(-12, 12)
        & ((frame.xy_dist_um / 1200.0) ** 2 + (frame.z_dist_um / 25000.0) ** 2).le(1.0)
        & np.isfinite(frame.effective_margin_um)
        & frame.effective_margin_um.le(1200.0)
    ).to_numpy(bool)


def shortlist(frame: pd.DataFrame, scores: np.ndarray, fraction=0.4) -> pd.DataFrame:
    """Retain the selected gate plus the highest ranked 40% outside it."""
    validate_candidates(frame)
    if not 0 <= fraction <= 1:
        raise ValueError("Outside fraction must be between 0 and 1")
    out = frame.reset_index(drop=True).copy()
    base = selected_gate(out)
    out["stage1_score"] = scores
    out["selected_gate_member"] = base.astype(np.int8)
    outside = np.flatnonzero(~base)
    ranks = np.full(len(out), -1, dtype=np.int32)
    order = np.lexsort(
        (
            out.survivor_track_id.to_numpy()[outside],
            out.vanished_track_id.to_numpy()[outside],
            out.candidate_frame.to_numpy()[outside],
            -np.asarray(scores)[outside],
        )
    )
    ranks[outside[order]] = np.arange(len(outside), dtype=np.int32)
    out["outside_rank"] = ranks
    keep = base | ((ranks >= 0) & (ranks < math.ceil(fraction * len(outside))))
    return out.loc[keep].reset_index(drop=True)


def score_static(frame: pd.DataFrame, model) -> pd.DataFrame:
    require_columns(frame, STATIC_FEATURES)
    out = frame.reset_index(drop=True).copy()
    score = model.predict_proba(feature_frame(out, STATIC_FEATURES))[:, 1]
    structural = logit_blend(out.stage1_score.to_numpy(float), score, 0.75)
    out["static_score"] = score
    out["structural_score"] = structural
    out["logit_blend_static_75_top3"] = structural
    for prefix, values in [("image", score), ("structural", structural)]:
        source = out[KEY_COLUMNS + GEOMETRY_COLUMNS].copy()
        source["rank_score"] = values
        ranked = attach_ranks(source, "rank_score")
        for rank in ["global", "pair", "vanished"]:
            out[f"{prefix}_{rank}_rank"] = ranked[f"{rank}_rank"].to_numpy(np.int32)
    return out


def image_mask(frame: pd.DataFrame, fraction=0.25) -> np.ndarray:
    return frame.image_global_rank.to_numpy(np.int64) < math.ceil(fraction * len(frame))


def scoring_union(static: pd.DataFrame, image: pd.DataFrame, recipe=None) -> pd.DataFrame:
    """Preserve paper missingness even when a complete image-feature bank exists."""
    recipe = recipe or paper_recipe()
    picked = image_mask(static, recipe["image_fraction"])
    retained = picked | static.structural_vanished_rank.lt(recipe["structural_max_vanished_rank"]).to_numpy()
    base = static.loc[retained].drop(columns=IMAGE_FEATURES, errors="ignore").reset_index(drop=True)
    chosen = static.loc[picked, KEY_COLUMNS].merge(
        image[KEY_COLUMNS + IMAGE_FEATURES], on=KEY_COLUMNS, how="left", validate="one_to_one", indicator=True
    )
    if chosen._merge.ne("both").any():
        raise ValueError("The image feature bank does not cover all selected image candidates")
    chosen = chosen.drop(columns="_merge")
    merged = base.merge(chosen, on=KEY_COLUMNS, how="left", sort=False, validate="one_to_one", indicator=True)
    merged["image_enriched"] = merged._merge.eq("both").astype(np.int8)
    result = add_derived_features(merged.drop(columns="_merge"))
    if not result[KEY_COLUMNS].equals(base[KEY_COLUMNS]):
        raise RuntimeError("Image merge changed candidate order")
    return result


def ensemble_scores(models, frame: pd.DataFrame) -> np.ndarray:
    require_columns(frame, ENHANCED_FEATURES)
    matrix = feature_frame(frame, ENHANCED_FEATURES)
    return np.mean([model.predict_proba(matrix)[:, 1] for model in models], axis=0)


def predict_full(
    full: pd.DataFrame, bundle: dict, *, scene_id: int = 0, base_scores: np.ndarray | None = None
) -> tuple[pd.DataFrame, dict]:
    if full.empty:
        return pd.DataFrame(columns=KEY_COLUMNS + GEOMETRY_COLUMNS + ["candidate_frame_raw", "score"]), {
            "candidates": 0,
            "events": 0,
        }
    validate_candidates(full)
    require_columns(full, ENHANCED_FEATURES)
    data = full.reset_index(drop=True).copy()
    data["synth_seed"] = scene_id
    # Inference does not consult labels. These fields only satisfy tensor construction.
    data["spatial_positive"] = False
    data["matched_truth_event_id"] = -1
    base = ensemble_scores(bundle["full_models"], data) if base_scores is None else np.asarray(base_scores)
    context = context_frame(data, base)
    rank, audit = relation_scores(
        context, bundle["relation_model"], bundle["relation_mean"], bundle["relation_std"], bundle["recipe"]
    )
    data["relation_attention_percentile"] = rank
    selected, policy = temporal_mode_predictions(
        data, seed=scene_id, method="uncorrected_score", config=bundle["recipe"]["source_policy"]
    )
    prediction = correct_predictions(
        selected, data, bundle["residual_models"], source_method="uncorrected_score", method="score"
    )
    source = data.iloc[prediction._source_index.to_numpy(np.int64)]
    prediction["candidate_frame_raw"] = source.candidate_frame.to_numpy(np.int64)
    for axis in "xyz":
        prediction[f"{axis}_um_raw"] = source[f"{axis}_um"].to_numpy(float)
    prediction["vanished_end_frame"] = source.vanished_end_frame.to_numpy(np.int64)
    prediction["base_score"] = base[prediction._source_index.to_numpy(np.int64)]
    prediction["image_enriched"] = source.image_enriched.to_numpy(np.int8)
    prediction["candidate_id"] = [
        f"{scene_id}:{int(v)}:{int(s)}:{int(f)}"
        for v, s, f in zip(
            prediction.vanished_track_id,
            prediction.survivor_track_id,
            prediction.candidate_frame_raw,
            strict=True,
        )
    ]
    prediction = prediction.sort_values(
        ["score", "candidate_frame_raw", "vanished_track_id", "survivor_track_id"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    prediction["review_rank"] = np.arange(1, len(prediction) + 1)
    return prediction, {"relation": audit, "policy": policy}

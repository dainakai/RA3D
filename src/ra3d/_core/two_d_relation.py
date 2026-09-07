"""Depth-free relation attention, event policy, and coordinate correction."""

from __future__ import annotations
import math
import time
from typing import Any, Sequence
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
from . import two_d_common as common


BASE_CONTEXT_FEATURES = ("score", "score_percentile", "structural_score", "time_offset_frame")

GROUP_SPECS = {
    "vanished": ["vanished_track_id"],
    "survivor": ["survivor_track_id"],
    "pair": ["vanished_track_id", "survivor_track_id"],
}

CONTEXT_FEATURES = tuple(
    list(BASE_CONTEXT_FEATURES)
    + [
        f"{prefix}_{suffix}"
        for prefix in GROUP_SPECS
        for suffix in (
            "max",
            "second",
            "top3_mean",
            "mean",
            "std",
            "log_count",
            "rank",
            "below_max",
            "frame_span",
        )
    ]
    + ["vanished_log_unique_survivors", "survivor_log_unique_vanished"]
)

RELATION_FEATURES = tuple(
    list(CONTEXT_FEATURES)
    + [
        "frame_from_top",
        "x_from_top",
        "y_from_top",
        "frame_from_mean",
        "x_from_mean",
        "y_from_mean",
        "same_survivor_as_top",
        "survivor_set_fraction",
    ]
)


def add_group_context(frame: pd.DataFrame, group_columns: list[str], prefix: str) -> None:
    group = frame.groupby(group_columns, sort=False)
    frame[f"{prefix}_max"] = group["score"].transform("max")
    frame[f"{prefix}_mean"] = group["score"].transform("mean")
    frame[f"{prefix}_std"] = group["score"].transform("std").fillna(0.0)
    frame[f"{prefix}_log_count"] = np.log1p(group["score"].transform("size").to_numpy(float))
    frame[f"{prefix}_frame_span"] = (
        group["candidate_frame"].transform("max") - group["candidate_frame"].transform("min")
    ).astype(float)
    ordered = frame.sort_values(
        group_columns + ["score", "candidate_frame", "_source_index"],
        ascending=[True] * len(group_columns) + [False, True, True],
        kind="mergesort",
    ).copy()
    ordered["_local_rank"] = ordered.groupby(group_columns, sort=False).cumcount()
    rank = ordered.set_index("_source_index")["_local_rank"]
    frame[f"{prefix}_rank"] = frame["_source_index"].map(rank).astype(np.int32)
    second_rows = ordered[ordered["_local_rank"].eq(1)]
    top3 = ordered[ordered["_local_rank"].lt(3)].groupby(group_columns, sort=False)["score"].mean()
    if len(group_columns) == 1:
        second = second_rows.set_index(group_columns[0])["score"]
        keys: Any = frame[group_columns[0]]
        frame[f"{prefix}_second"] = keys.map(second)
        frame[f"{prefix}_top3_mean"] = keys.map(top3)
    else:
        second = second_rows.set_index(group_columns)["score"]
        keys = pd.MultiIndex.from_frame(frame[group_columns])
        frame[f"{prefix}_second"] = second.reindex(keys).to_numpy(float)
        frame[f"{prefix}_top3_mean"] = top3.reindex(keys).to_numpy(float)
    frame[f"{prefix}_second"] = frame[f"{prefix}_second"].fillna(frame[f"{prefix}_max"])
    frame[f"{prefix}_top3_mean"] = frame[f"{prefix}_top3_mean"].fillna(frame[f"{prefix}_max"])
    frame[f"{prefix}_below_max"] = frame[f"{prefix}_max"] - frame["score"]


def context_frame(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    columns = [
        "synth_seed",
        *common.KEY_COLUMNS,
        "vanished_end_frame",
        "time_offset_frame",
        "x_um",
        "y_um",
        "z_um",
        "structural_score",
        "relation_positive",
        "matched_truth_event_id",
    ]
    output = frame[[column for column in columns if column in frame]].copy()
    output["_source_index"] = np.arange(len(output), dtype=np.int64)
    output["score"] = np.asarray(scores, dtype=np.float64)
    output["score_percentile"] = common.percentile_score(scores)
    for prefix, group_columns in GROUP_SPECS.items():
        add_group_context(output, group_columns, prefix)
    vanished = output.groupby("vanished_track_id", sort=False)
    survivor = output.groupby("survivor_track_id", sort=False)
    output["vanished_log_unique_survivors"] = np.log1p(
        vanished["survivor_track_id"].transform("nunique").to_numpy(float)
    )
    output["survivor_log_unique_vanished"] = np.log1p(
        survivor["vanished_track_id"].transform("nunique").to_numpy(float)
    )
    return output


class RelationAttentionRanker(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        hidden_size = 48
        self.input = nn.Sequential(
            nn.Linear(feature_count, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.attention = nn.MultiheadAttention(hidden_size, 4, dropout=0.0, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.ReLU(), nn.Linear(hidden_size * 2, hidden_size)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.input(features)
        (attended, _weights) = self.attention(
            encoded, encoded, encoded, key_padding_mask=~mask, need_weights=False
        )
        encoded = self.attention_norm(encoded + attended)
        encoded = self.output_norm(encoded + self.feed_forward(encoded))
        valid = mask.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        top = encoded[:, 0, :]
        group = torch.cat([mean, top], dim=-1)
        expanded = group.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        logits = self.scorer(torch.cat([encoded, expanded], dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, -1000000000.0)


def relation_columns(context: pd.DataFrame, selected: np.ndarray) -> np.ndarray:
    frame = context.loc[selected, "candidate_frame"].to_numpy(np.float32)
    x_value = context.loc[selected, "x_um"].to_numpy(np.float32)
    y_value = context.loc[selected, "y_um"].to_numpy(np.float32)
    survivor = context.loc[selected, "survivor_track_id"].to_numpy(np.int64)
    (unique, counts) = np.unique(survivor, return_counts=True)
    count_by_survivor = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    fraction = np.asarray(
        [count_by_survivor[int(value)] / len(selected) for value in survivor], dtype=np.float32
    )
    return np.column_stack(
        [
            frame - frame[0],
            x_value - x_value[0],
            y_value - y_value[0],
            frame - frame.mean(),
            x_value - x_value.mean(),
            y_value - y_value.mean(),
            survivor == survivor[0],
            fraction,
        ]
    ).astype(np.float32)


def build_set_arrays(
    contexts: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    feature_rows = []
    mask_rows = []
    label_rows = []
    source_rows = []
    positive_groups = 0
    retained_positive_groups = 0
    for context_id, context in enumerate(contexts):
        base = np.nan_to_num(
            context.loc[:, list(CONTEXT_FEATURES)].to_numpy(np.float32),
            nan=0.0,
            posinf=1000000.0,
            neginf=-1000000.0,
        )
        labels = context["relation_positive"].to_numpy(bool)
        scores = context["score"].to_numpy(float)
        for indices in context.groupby("vanished_track_id", sort=False).indices.values():
            local = np.asarray(indices, dtype=np.int64)
            has_positive = bool(labels[local].any())
            positive_groups += int(has_positive)
            selected = local[np.argsort(-scores[local], kind="stable")[:20]]
            retained_positive_groups += int(has_positive and bool(labels[selected].any()))
            features = np.zeros((20, len(RELATION_FEATURES)), dtype=np.float32)
            mask = np.zeros(20, dtype=bool)
            local_labels = np.zeros(20, dtype=bool)
            sources = np.full((20, 2), -1, dtype=np.int64)
            count = len(selected)
            features[:count, : len(CONTEXT_FEATURES)] = base[selected]
            features[:count, len(CONTEXT_FEATURES) :] = relation_columns(context, selected)
            mask[:count] = True
            local_labels[:count] = labels[selected]
            sources[:count, 0] = context_id
            sources[:count, 1] = selected
            feature_rows.append(features)
            mask_rows.append(mask)
            label_rows.append(local_labels)
            source_rows.append(sources)
    return (
        np.stack(feature_rows),
        np.stack(mask_rows),
        np.stack(label_rows),
        np.stack(source_rows),
        {
            "sets": int(len(feature_rows)),
            "positive_groups": int(positive_groups),
            "positive_groups_with_top20_positive": int(retained_positive_groups),
            "positive_top20_coverage": float(
                retained_positive_groups / positive_groups if positive_groups else 0.0
            ),
        },
    )


def relation_weights(
    contexts: list[pd.DataFrame], labels: np.ndarray, mask: np.ndarray, sources: np.ndarray
) -> np.ndarray:
    weights = np.zeros(labels.shape, dtype=np.float32)
    for context_id, context in enumerate(contexts):
        positions = np.argwhere(mask & (sources[:, :, 0] == context_id))
        indices = sources[positions[:, 0], positions[:, 1], 1]
        local_labels = labels[positions[:, 0], positions[:, 1]]
        positives = positions[local_labels]
        negatives = positions[~local_labels]
        if len(positives):
            event_ids = context.iloc[indices[local_labels]]["matched_truth_event_id"].to_numpy()
            event_counts = pd.Series(event_ids).value_counts()
            positive_weights = np.asarray(
                [1.0 / float(event_counts.loc[event]) for event in event_ids], dtype=np.float64
            )
            positive_weights /= positive_weights.sum()
            weights[positives[:, 0], positives[:, 1]] = positive_weights
        if len(negatives):
            weights[negatives[:, 0], negatives[:, 1]] = 1.0 / len(negatives)
    observed = weights[mask]
    weights /= float(observed.mean())
    return weights


def fit_attention(
    contexts: list[pd.DataFrame], requested_device: str
) -> tuple[RelationAttentionRanker, np.ndarray, np.ndarray, dict[str, Any]]:
    (features, mask, labels, sources, audit) = build_set_arrays(contexts)
    weights = relation_weights(contexts, labels, mask, sources)
    observed = features[mask]
    mean = observed.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = observed.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1e-06, std, 1.0).astype(np.float32)
    features = ((features - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    features[~mask] = 0.0
    device = torch.device(requested_device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        device = torch.device("cpu")
    torch.manual_seed(270901)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(270901)
    model = RelationAttentionRanker(len(RELATION_FEATURES)).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(270901)
    losses = []
    started = time.perf_counter()
    model.train()
    for _epoch in range(12):
        order = torch.randperm(len(features), generator=generator).numpy()
        weighted_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), 192):
            indices = order[start : start + 192]
            x = torch.from_numpy(features[indices]).to(device)
            valid = torch.from_numpy(mask[indices]).to(device)
            target = torch.from_numpy(labels[indices].astype(np.float32)).to(device)
            weight = torch.from_numpy(weights[indices]).to(device)
            logits = model(x, valid)
            raw = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            denominator = weight.sum().clamp_min(1e-12)
            loss = (raw * weight).sum() / denominator
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted_loss += float((raw * weight).sum().detach().cpu())
            total_weight += float(denominator.detach().cpu())
        losses.append(weighted_loss / total_weight)
    audit.update(
        {
            "device": str(device),
            "feature_count": int(len(RELATION_FEATURES)),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_gpu_memory_gb": float(
                torch.cuda.max_memory_allocated() / 1024.0**3 if device.type == "cuda" else 0.0
            ),
        }
    )
    return (model, mean, std, audit)


def attention_scores(
    context: pd.DataFrame, model: RelationAttentionRanker, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    (features, mask, _labels, sources, audit) = build_set_arrays([context])
    features = ((features - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    features[~mask] = 0.0
    device = next(model.parameters()).device
    logits = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 512):
            end = start + 512
            logits.append(
                model(
                    torch.from_numpy(features[start:end]).to(device),
                    torch.from_numpy(mask[start:end]).to(device),
                )
                .cpu()
                .numpy()
            )
    logits_array = np.concatenate(logits, axis=0)
    base = context["score"].to_numpy(float)
    learned = base.copy()
    selected_rows = 0
    for set_index in range(len(features)):
        valid = np.flatnonzero(mask[set_index])
        indices = sources[set_index, valid, 1]
        learned[indices] = 1.0 / (1.0 + np.exp(-np.clip(logits_array[set_index, valid], -40.0, 40.0)))
        selected_rows += len(valid)
    blended = common.logit_blend(base, learned, 0.15)
    audit["selected_rows"] = int(selected_rows)
    audit["mean_absolute_probability_change"] = float(np.mean(np.abs(blended - base)))
    return (common.percentile_score(blended), audit)


def coordinate_nms_2d(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["_policy_score", "candidate_frame", "_source_index"], ascending=[False, True, True], kind="mergesort"
    )
    kept = []
    by_frame: dict[int, list[tuple[float, float]]] = {}
    for row in ordered.itertuples():
        event_frame = int(row.candidate_frame)
        x_value = float(row.x_um)
        y_value = float(row.y_um)
        conflict = False
        for nearby_frame in range(event_frame - 2, event_frame + 3):
            for prior_x, prior_y in by_frame.get(nearby_frame, []):
                if math.hypot(x_value - prior_x, y_value - prior_y) <= 250.0:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        kept.append(int(row.Index))
        by_frame.setdefault(event_frame, []).append((x_value, y_value))
    return frame.loc[kept].copy()


def policy_predictions(
    frame: pd.DataFrame, score: np.ndarray, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = [*common.KEY_COLUMNS, "vanished_end_frame", "time_offset_frame", "x_um", "y_um", "z_um"]
    work = frame[columns].copy()
    work["identity_candidate_frame"] = work["candidate_frame"].to_numpy(np.int64)
    work["_source_index"] = np.arange(len(work), dtype=np.int64)
    work["_policy_score"] = np.asarray(score, dtype=float)
    ordered = work.sort_values(
        ["vanished_track_id", "survivor_track_id", "candidate_frame", "_policy_score", "_source_index"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).copy()
    difference = ordered.groupby(["vanished_track_id", "survivor_track_id"], sort=False)[
        "candidate_frame"
    ].diff()
    ordered["_new_temporal_mode"] = (difference.isna() | difference.gt(10)).astype(np.int32)
    ordered["_temporal_mode"] = (
        ordered.groupby(["vanished_track_id", "survivor_track_id"], sort=False)["_new_temporal_mode"]
        .cumsum()
        .astype(np.int32)
    )
    representatives = (
        ordered.sort_values(
            ["_policy_score", "candidate_frame", "vanished_track_id", "survivor_track_id", "_source_index"],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        )
        .drop_duplicates(["vanished_track_id", "survivor_track_id", "_temporal_mode"], keep="first")
        .copy()
    )
    representatives = representatives.sort_values(
        ["_policy_score", "candidate_frame", "_source_index"], ascending=[False, True, True], kind="mergesort"
    )
    representatives["_lineage_rank"] = representatives.groupby("vanished_track_id", sort=False).cumcount()
    capped = representatives[representatives["_lineage_rank"] < 5].copy()
    selected = coordinate_nms_2d(capped)
    probability = np.clip(selected["_policy_score"].to_numpy(float), 1e-12, 1.0 - 1e-12)
    logit = np.log(probability / (1.0 - probability))
    selected["relation_attention_2d_uncorrected"] = 1.0 / (
        1.0 + np.exp(-np.clip(logit - selected["_lineage_rank"].to_numpy(float), -40.0, 40.0))
    )
    selected["synth_seed"] = int(seed)
    audit = {
        "seed": int(seed),
        "input_candidates": int(len(work)),
        "temporal_pair_modes": int(len(representatives)),
        "post_vanished_cap": int(len(capped)),
        "post_coordinate_suppression": int(len(selected)),
    }
    return (selected, audit)


def fit_residual_models(
    frames: dict[int, pd.DataFrame], train_seeds: Sequence[int], truths: dict[int, pd.DataFrame]
) -> tuple[dict[str, HistGradientBoostingRegressor], dict[str, Any]]:
    feature_parts = []
    targets = {name: [] for name in ("frame_delta", "x_delta", "y_delta")}
    weight_parts = []
    for seed in train_seeds:
        frame = frames[int(seed)]
        selected = frame.loc[frame["identity_training_selected"]].copy()
        truth = truths[int(seed)].reset_index(drop=True)
        truth_indices = selected["assigned_truth_index"].to_numpy(np.int64)
        targets["frame_delta"].append(
            truth.loc[truth_indices, "frame"].to_numpy(float) - selected["candidate_frame"].to_numpy(float)
        )
        targets["x_delta"].append(
            truth.loc[truth_indices, "x_um"].to_numpy(float) - selected["x_um"].to_numpy(float)
        )
        targets["y_delta"].append(
            truth.loc[truth_indices, "y_um"].to_numpy(float) - selected["y_um"].to_numpy(float)
        )
        feature_parts.append(selected)
        weight_parts.append(selected["identity_event_factor"].to_numpy(float))
    training = pd.concat(feature_parts, ignore_index=True, sort=False)
    target_values = {name: np.concatenate(parts) for (name, parts) in targets.items()}
    weights = np.concatenate(weight_parts)
    weights /= weights.mean()
    matrix = common.feature_frame(training, list(common.TWO_D_FEATURES))
    models = {}
    target_audits = {}
    started = time.perf_counter()
    for offset, name in enumerate(("frame_delta", "x_delta", "y_delta")):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=270100 + offset,
        )
        local_started = time.perf_counter()
        model.fit(matrix, target_values[name], sample_weight=weights)
        fitted = model.predict(matrix)
        models[name] = model
        target_audits[name] = {
            "training_rmse": float(np.sqrt(np.average((fitted - target_values[name]) ** 2, weights=weights))),
            "elapsed_seconds": float(time.perf_counter() - local_started),
        }
    return (
        models,
        {
            "train_seeds": [int(seed) for seed in train_seeds],
            "training_rows": int(len(training)),
            "target_audits": target_audits,
            "elapsed_seconds": float(time.perf_counter() - started),
        },
    )


def correct_predictions(
    prediction: pd.DataFrame, full_frame: pd.DataFrame, models: dict[str, HistGradientBoostingRegressor]
) -> pd.DataFrame:
    source_indices = prediction["_source_index"].to_numpy(np.int64)
    selected = full_frame.iloc[source_indices]
    matrix = common.feature_frame(selected, list(common.TWO_D_FEATURES))
    correction = {name: model.predict(matrix) for (name, model) in models.items()}
    output = prediction.copy()
    output["candidate_frame"] = np.rint(
        output["candidate_frame"].to_numpy(float) + np.clip(correction["frame_delta"], -5.0, 5.0)
    ).astype(np.int64)
    output["x_um"] = output["x_um"].to_numpy(float) + np.clip(correction["x_delta"], -350.0, 350.0)
    output["y_um"] = output["y_um"].to_numpy(float) + np.clip(correction["y_delta"], -350.0, 350.0)
    output["z_um"] = 0.0
    output["relation_attention_2d"] = output["relation_attention_2d_uncorrected"].to_numpy(float)
    output["method"] = "relation_attention_2d"
    return output

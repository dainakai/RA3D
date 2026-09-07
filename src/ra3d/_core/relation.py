"""Numerical kernels derived from run_relation_attention.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
import time
import torch
from torch import nn
from . import common as campaign
from .context import CONTEXT_FEATURES, blend_scores


RELATION_FEATURES = tuple(
    list(CONTEXT_FEATURES)
    + [
        "frame_from_top",
        "x_from_top",
        "y_from_top",
        "z_from_top",
        "frame_from_mean",
        "x_from_mean",
        "y_from_mean",
        "z_from_mean",
        "same_survivor_as_top",
        "survivor_set_fraction",
    ]
)


class RelationAttentionRanker(nn.Module):
    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.attention = nn.MultiheadAttention(
            hidden_size,
            attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.input(features)
        attended, _weights = self.attention(
            encoded,
            encoded,
            encoded,
            key_padding_mask=~mask,
            need_weights=False,
        )
        encoded = self.attention_norm(encoded + attended)
        encoded = self.output_norm(encoded + self.feed_forward(encoded))
        valid = mask.unsqueeze(-1).to(encoded.dtype)
        mean = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        top = encoded[:, 0, :]
        group = torch.cat([mean, top], dim=-1)
        expanded = group.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        logits = self.scorer(torch.cat([encoded, expanded], dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, -1.0e9)


def finite_context_features(context: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(
        context.loc[:, list(CONTEXT_FEATURES)].to_numpy(np.float32),
        nan=0.0,
        posinf=1.0e6,
        neginf=-1.0e6,
    )


def relation_columns(context: pd.DataFrame, selected: np.ndarray) -> np.ndarray:
    frame = context.loc[selected, "candidate_frame"].to_numpy(np.float32)
    x_value = context.loc[selected, "x_um"].to_numpy(np.float32)
    y_value = context.loc[selected, "y_um"].to_numpy(np.float32)
    z_value = context.loc[selected, "z_um"].to_numpy(np.float32)
    survivor = context.loc[selected, "survivor_track_id"].to_numpy(np.int64)
    unique, counts = np.unique(survivor, return_counts=True)
    count_by_survivor = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    survivor_fraction = np.asarray(
        [count_by_survivor[int(value)] / len(selected) for value in survivor],
        dtype=np.float32,
    )
    return np.column_stack(
        [
            frame - frame[0],
            x_value - x_value[0],
            y_value - y_value[0],
            z_value - z_value[0],
            frame - frame.mean(),
            x_value - x_value.mean(),
            y_value - y_value.mean(),
            z_value - z_value.mean(),
            survivor == survivor[0],
            survivor_fraction,
        ]
    ).astype(np.float32)


def build_set_arrays(
    contexts: list[pd.DataFrame],
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    feature_rows = []
    mask_rows = []
    label_rows = []
    source_rows = []
    total_groups = 0
    positive_groups = 0
    retained_positive_groups = 0
    for context_id, context in enumerate(contexts):
        base_features = finite_context_features(context)
        labels = context["spatial_positive"].to_numpy(bool)
        score = context["score"].to_numpy(float)
        for indices in context.groupby("vanished_track_id", sort=False).indices.values():
            total_groups += 1
            local = np.asarray(indices, dtype=np.int64)
            has_positive = bool(labels[local].any())
            positive_groups += int(has_positive)
            selected = local[np.argsort(-score[local], kind="stable")[:top_k]]
            selected_has_positive = bool(labels[selected].any())
            retained_positive_groups += int(has_positive and selected_has_positive)
            feature_row = np.zeros((top_k, len(RELATION_FEATURES)), dtype=np.float32)
            mask_row = np.zeros(top_k, dtype=bool)
            label_row = np.zeros(top_k, dtype=bool)
            source_row = np.full((top_k, 2), -1, dtype=np.int64)
            count = len(selected)
            feature_row[:count, : len(CONTEXT_FEATURES)] = base_features[selected]
            feature_row[:count, len(CONTEXT_FEATURES) :] = relation_columns(context, selected)
            mask_row[:count] = True
            label_row[:count] = labels[selected]
            source_row[:count, 0] = context_id
            source_row[:count, 1] = selected
            feature_rows.append(feature_row)
            mask_rows.append(mask_row)
            label_rows.append(label_row)
            source_rows.append(source_row)
    return (
        np.stack(feature_rows),
        np.stack(mask_rows),
        np.stack(label_rows),
        np.stack(source_rows),
        {
            "total_groups": int(total_groups),
            "positive_groups": int(positive_groups),
            "positive_groups_with_topk_positive": int(retained_positive_groups),
            "positive_topk_coverage": float(
                retained_positive_groups / positive_groups if positive_groups else 0.0
            ),
            "output_sets": int(len(feature_rows)),
        },
    )


def event_balanced_weights(
    contexts: list[pd.DataFrame],
    labels: np.ndarray,
    mask: np.ndarray,
    sources: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    weights = np.zeros(labels.shape, dtype=np.float32)
    audits = []
    for context_id, context in enumerate(contexts):
        positions = np.argwhere(mask & (sources[:, :, 0] == context_id))
        local_indices = sources[positions[:, 0], positions[:, 1], 1]
        local_labels = labels[positions[:, 0], positions[:, 1]]
        positive_positions = positions[local_labels]
        negative_positions = positions[~local_labels]
        positive_indices = local_indices[local_labels]
        event_ids = context.iloc[positive_indices]["matched_truth_event_id"].to_numpy()
        if len(positive_positions):
            event_counts = pd.Series(event_ids).value_counts(dropna=False)
            positive_weight = np.asarray(
                [1.0 / float(event_counts.loc[event_id]) for event_id in event_ids],
                dtype=np.float64,
            )
            positive_weight /= positive_weight.sum()
            weights[positive_positions[:, 0], positive_positions[:, 1]] = positive_weight
        if len(negative_positions):
            weights[negative_positions[:, 0], negative_positions[:, 1]] = 1.0 / len(negative_positions)
        audits.append(
            {
                "seed": int(context["synth_seed"].iloc[0]),
                "selected_rows": int(len(positions)),
                "selected_positive_rows": int(len(positive_positions)),
                "selected_negative_rows": int(len(negative_positions)),
                "selected_positive_events": int(pd.Series(event_ids).nunique()),
            }
        )
    observed = weights[mask]
    weights /= float(observed.mean())
    return weights, audits


def normalization(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = features[mask]
    mean = observed.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = observed.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std > 1.0e-6, std, 1.0).astype(np.float32)
    return mean, std


def normalize_features(
    features: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    output = (features - mean[None, None, :]) / std[None, None, :]
    output[~mask] = 0.0
    return output.astype(np.float32, copy=False)


def choose_device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested.type == "cuda" and requested.index is not None:
        if requested.index >= torch.cuda.device_count():
            return torch.device("cpu")
    return requested


def fit_model(
    contexts: list[pd.DataFrame],
    config: dict[str, Any],
) -> tuple[RelationAttentionRanker, np.ndarray, np.ndarray, dict[str, Any]]:
    features, mask, labels, sources, set_audit = build_set_arrays(contexts, top_k=int(config["set_top_k"]))
    weights, weight_audits = event_balanced_weights(contexts, labels, mask, sources)
    mean, std = normalization(features, mask)
    features = normalize_features(features, mask, mean, std)
    device = choose_device(str(config["device"]))
    torch.manual_seed(int(config["random_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["random_seed"]))
        torch.cuda.reset_peak_memory_stats(device)
    model = RelationAttentionRanker(
        len(RELATION_FEATURES),
        int(config["hidden_size"]),
        int(config["attention_heads"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    order_generator = torch.Generator().manual_seed(int(config["random_seed"]))
    batch_size = int(config["batch_size"])
    losses = []
    started = time.perf_counter()
    model.train()
    for _epoch in range(int(config["epochs"])):
        order = torch.randperm(len(features), generator=order_generator).numpy()
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = torch.from_numpy(features[indices]).to(device)
            valid = torch.from_numpy(mask[indices]).to(device)
            target = torch.from_numpy(labels[indices].astype(np.float32)).to(device)
            weight = torch.from_numpy(weights[indices]).to(device)
            logits = model(x, valid)
            raw_loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            denominator = weight.sum().clamp_min(1.0e-12)
            loss = (raw_loss * weight).sum() / denominator
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float((raw_loss * weight).sum().detach().cpu())
            epoch_weight += float(denominator.detach().cpu())
        losses.append(epoch_loss / epoch_weight)
    audit = {
        **set_audit,
        "weight_audits": weight_audits,
        "device": str(device),
        "epochs": int(config["epochs"]),
        "initial_epoch_loss": float(losses[0]),
        "final_epoch_loss": float(losses[-1]),
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_gpu_memory_gb": float(
            torch.cuda.max_memory_allocated(device) / (1024.0**3) if device.type == "cuda" else 0.0
        ),
    }
    return model, mean, std, audit


def relation_scores(
    context: pd.DataFrame,
    model: RelationAttentionRanker,
    mean: np.ndarray,
    std: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    features, mask, _labels, sources, set_audit = build_set_arrays([context], top_k=int(config["set_top_k"]))
    features = normalize_features(features, mask, mean, std)
    device = next(model.parameters()).device
    logit_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), int(config["inference_batch_size"])):
            end = start + int(config["inference_batch_size"])
            logit_parts.append(
                model(
                    torch.from_numpy(features[start:end]).to(device),
                    torch.from_numpy(mask[start:end]).to(device),
                )
                .cpu()
                .numpy()
            )
    logits = np.concatenate(logit_parts, axis=0)
    base = context["score"].to_numpy(float)
    learned = base.copy()
    selected_rows = 0
    for set_index in range(len(features)):
        valid_positions = np.flatnonzero(mask[set_index])
        source_indices = sources[set_index, valid_positions, 1]
        learned[source_indices] = 1.0 / (
            1.0 + np.exp(-np.clip(logits[set_index, valid_positions], -40.0, 40.0))
        )
        selected_rows += len(valid_positions)
    blended = blend_scores(base, learned, float(config["relation_logit_weight"]))
    set_audit.update(
        {
            "selected_rows": int(selected_rows),
            "mean_absolute_probability_change": float(np.mean(np.abs(blended - base))),
        }
    )
    return campaign.percentile_score(blended), set_audit

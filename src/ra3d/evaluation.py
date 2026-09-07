"""Scene-separated, maximum one-to-one event precision–recall evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._core.matching import candidate_truth_adjacency
from .io import require_columns, require_integer


def event_curve(
    predictions: dict,
    truths: dict,
    *,
    score_column="score",
    projected_xy=True,
    frame_tolerance=5,
    xy_tolerance_um=350.0,
    z_tolerance_um=15000.0,
):
    """Pool scores across scenes, never matching events across acquisition boundaries.

    The paper's projected-XY protocol uses corrected frame/x/y and no z gate.
    Labels for training use a different, three-dimensional association rule.
    """
    if set(predictions) != set(truths):
        raise ValueError("Prediction and truth scene sets must match")
    total_truth = sum(len(t) for t in truths.values())
    if total_truth == 0:
        raise ValueError("Event recall/AP requires at least one truth event")
    adjacency = {}
    scored = []
    truth_to_prediction = {}
    prediction_to_truth = {}
    for scene in sorted(truths):
        pred = predictions[scene].reset_index(drop=True).copy()
        truth = truths[scene].reset_index(drop=True).copy()
        require_columns(pred, [score_column, "candidate_frame", "x_um", "y_um"], finite=True)
        require_columns(truth, ["frame", "x_um", "y_um"], finite=True)
        require_integer(pred, ["candidate_frame"])
        require_integer(truth, ["frame"])
        if "collision_event_id" not in truth:
            truth["collision_event_id"] = np.arange(len(truth))
        if projected_xy:
            pred["z_um"] = 0.0
            truth["z_um"] = 0.0
        else:
            require_columns(pred, ["z_um"], finite=True)
            require_columns(truth, ["z_um"], finite=True)
        local = candidate_truth_adjacency(
            pred,
            truth,
            frame_tolerance=frame_tolerance,
            xy_tolerance_um=xy_tolerance_um,
            z_tolerance_um=z_tolerance_um,
        )
        for index, neighbors in local.items():
            adjacency[(scene, index)] = [(scene, j) for j in neighbors]
        scored.extend((float(score), scene, i) for i, score in enumerate(pred[score_column]))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    rows = []
    for rank, (score, scene, index) in enumerate(scored, start=1):
        root = (scene, index)
        queue = [root]
        seen_predictions = {root}
        seen_truth = set()
        parent = {}
        endpoint = None
        for prediction in queue:
            for truth in adjacency.get(prediction, []):
                if truth in seen_truth:
                    continue
                seen_truth.add(truth)
                previous = truth_to_prediction.get(truth)
                if previous is None:
                    endpoint = (prediction, truth)
                    break
                if previous not in seen_predictions:
                    seen_predictions.add(previous)
                    parent[previous] = (prediction, truth)
                    queue.append(previous)
            if endpoint is not None:
                break
        if endpoint is not None:
            prediction, truth = endpoint
            while True:
                truth_to_prediction[truth] = prediction
                prediction_to_truth[prediction] = truth
                if prediction == root:
                    break
                prediction, truth = parent[prediction]
        if rank == len(scored) or scored[rank][0] != score:
            matched = len(truth_to_prediction)
            precision = matched / rank
            recall = matched / total_truth
            rows.append(
                {
                    "threshold": score,
                    "predicted_events": rank,
                    "matched_events": matched,
                    "truth_events": total_truth,
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                }
            )
    if not rows:
        rows = [
            dict(
                threshold=1.0,
                predicted_events=0,
                matched_events=0,
                truth_events=total_truth,
                precision=1.0,
                recall=0.0,
                f1=0.0,
            )
        ]
    curve = pd.DataFrame(rows)
    recall = curve.recall.to_numpy(float)
    envelope = np.maximum.accumulate(curve.precision.to_numpy(float)[::-1])[::-1]
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * envelope))
    summary = {
        "event_ap": ap,
        "truth_events": total_truth,
        "predicted_events": len(scored),
        "maximum_recall": float(recall.max()),
        "protocol": "projected_xy" if projected_xy else "3d",
        "frame_tolerance": frame_tolerance,
        "xy_tolerance_um": xy_tolerance_um,
        "z_tolerance_um": None if projected_xy else z_tolerance_um,
        "one_to_one_matching": "maximum",
        "score_column": score_column,
    }
    for floor in [0.8, 0.9]:
        eligible = curve.loc[curve.precision >= floor]
        summary[f"operating_at_precision_{floor:.2f}"] = (
            None
            if eligible.empty
            else eligible.sort_values(["recall", "precision", "threshold"], ascending=False, kind="stable")
            .iloc[0]
            .to_dict()
        )
    return curve, summary

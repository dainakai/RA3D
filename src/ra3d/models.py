"""Portable model directories using skops and safetensors instead of pickle."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import sklearn
import skops.io as sio
from safetensors.torch import save_file, load_file

from ._core.relation import RelationAttentionRanker
from .config import ENHANCED_FEATURES, STAGE1_FEATURES, STATIC_FEATURES
from .io import sha256, write_json

TRUSTED_SKLEARN_TYPES = {
    "sklearn._loss.link.Interval",
    "sklearn._loss.link.LogitLink",
    "sklearn._loss.link.IdentityLink",
    "sklearn._loss.loss.HalfBinomialLoss",
    "sklearn._loss.loss.HalfSquaredError",
    "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "_loss.CyHalfBinomialLoss",
    "_loss.CyHalfSquaredError",
}


def save_model(bundle: dict, output: str | Path, *, provenance: dict) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tree_keys = ["stage1_model", "static_model", "full_models", "residual_models"]
    sio.dump({key: bundle[key] for key in tree_keys}, output / "trees.skops")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in bundle["relation_model"].state_dict().items()},
        output / "relation.safetensors",
    )
    np.savez(output / "normalization.npz", mean=bundle["relation_mean"], std=bundle["relation_std"])
    names = ["trees.skops", "relation.safetensors", "normalization.npz"]
    write_json(
        {
            "schema_version": 1,
            "sklearn_version": sklearn.__version__,
            "recipe": {**bundle["recipe"], "device": "cpu"},
            "features": ENHANCED_FEATURES,
            "stage1_features": STAGE1_FEATURES,
            "static_features": STATIC_FEATURES,
            "files": {name: sha256(output / name) for name in names},
            "provenance": provenance,
        },
        output / "model.json",
    )
    return output


def load_model(source: str | Path, *, device="cpu") -> dict:
    source = Path(source)
    config = json.loads((source / "model.json").read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported RA3D model schema")
    if config["sklearn_version"] != sklearn.__version__:
        raise ValueError(
            f"This model requires scikit-learn {config['sklearn_version']}; found {sklearn.__version__}"
        )
    if (
        config["features"] != ENHANCED_FEATURES
        or config["stage1_features"] != STAGE1_FEATURES
        or config["static_features"] != STATIC_FEATURES
    ):
        raise ValueError("Model feature names/order do not match this RA3D version")
    for name in ["trees.skops", "relation.safetensors", "normalization.npz"]:
        if sha256(source / name) != config["files"][name]:
            raise ValueError(f"Model checksum mismatch: {name}")
    untrusted = set(sio.get_untrusted_types(file=source / "trees.skops"))
    if not untrusted <= TRUSTED_SKLEARN_TYPES:
        raise ValueError(f"Unsupported model object types: {sorted(untrusted - TRUSTED_SKLEARN_TYPES)}")
    result = sio.load(source / "trees.skops", trusted=sorted(untrusted))
    recipe = {**config["recipe"], "device": device}
    model = RelationAttentionRanker(46, recipe["hidden_size"], recipe["attention_heads"])
    model.load_state_dict(load_file(source / "relation.safetensors"))
    model.to(device).eval()
    with np.load(source / "normalization.npz", allow_pickle=False) as norm:
        result.update(relation_mean=norm["mean"], relation_std=norm["std"])
    result.update(relation_model=model, recipe=recipe, provenance=config["provenance"])
    return result

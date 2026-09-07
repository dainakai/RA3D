import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ra3d.config import ENHANCED_FEATURES, STAGE1_FEATURES, STATIC_FEATURES, paper_recipe
from ra3d.models import save_model, load_model
from ra3d._core.relation import RelationAttentionRanker
from ra3d.training import TrainingCheckpoint, training_rows, sample_weights


def test_safe_model_roundtrip_and_checksum_rejection(tmp_path):
    rng = np.random.default_rng(6)
    x = pd.DataFrame(rng.normal(size=(32, 92)), columns=ENHANCED_FEATURES)
    y = np.arange(32) % 2

    def classifier(columns):
        return HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=2, early_stopping=False).fit(
            x[columns], y
        )

    model = classifier(ENHANCED_FEATURES)
    residual = {
        axis + "_delta": HistGradientBoostingRegressor(
            max_iter=2, min_samples_leaf=2, early_stopping=False
        ).fit(x, np.zeros(32))
        for axis in ["frame", "x", "y", "z"]
    }
    bundle = {
        "stage1_model": classifier(STAGE1_FEATURES),
        "static_model": classifier(STATIC_FEATURES),
        "full_models": [model] * 3,
        "residual_models": residual,
        "relation_model": RelationAttentionRanker(46, 48, 4),
        "relation_mean": np.zeros(46, dtype=np.float32),
        "relation_std": np.ones(46, dtype=np.float32),
        "recipe": {**paper_recipe(), "device": "cuda:7"},
    }
    save_model(bundle, tmp_path, provenance={"test": True})
    restored = load_model(tmp_path, device="cpu")
    assert restored["recipe"]["device"] == "cpu"
    np.testing.assert_array_equal(model.predict_proba(x), restored["full_models"][0].predict_proba(x))
    with (tmp_path / "relation.safetensors").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path)


def test_training_excludes_ambiguous_spatial_positive_and_balances_events():
    part = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "spatial_positive": [1, 1, 1, 0],
            "identity_training_selected": [True, True, False, False],
            "identity_event_factor": [0.5, 0.5, 0.0, 0.0],
        }
    )
    selected = training_rows([part], ["feature"])
    assert selected.feature.tolist() == [1.0, 2.0, 4.0]
    weights = sample_weights(selected)
    assert weights[:2].sum() == weights[2]
    assert weights.mean() == 1.0


def test_checkpoint_rejects_changed_training_inputs(tmp_path):
    TrainingCheckpoint(tmp_path, {"data": "hash-a"})
    with pytest.raises(ValueError, match="inputs or settings differ"):
        TrainingCheckpoint(tmp_path, {"data": "hash-b"})

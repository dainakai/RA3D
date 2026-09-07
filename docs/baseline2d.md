# Depth-free 2D counterpart

The 2D method uses 13 screening features, 36 additional trajectory features (49 total), and 41 relation inputs. It excludes Z, Z-dependent derived features, boundary-depth scores and reconstructed-image features. It is a separately trained comparison method.

## Refit and evaluate

```bash
ra3d download --tier baseline2d --tier labels --output data
python -m ra3d.baseline2d train --data data --output models/2d --device cpu
python -m ra3d.baseline2d predict \
  --features data/baseline2d/validation/seed260745/full.parquet \
  --model models/2d --scene-id 260745 --output runs/2d-terminal.parquet
ra3d evaluate --predictions runs/2d-terminal.parquet \
  --truth data/labels/validation/seed260745/collision_events.parquet \
  --output runs/2d-terminal-evaluation
```

The default fit recomputes scene OOF full-stage scores. `--reuse-scores` explicitly reuses the published arrays, recording their hashes and preserving row order. The released `models/baseline2d` bundle was refitted with that explicit reuse; independently refitted terminal base scores matched bit for bit, and CPU relation training/inference reproduced AP **0.8993428951464474**.

Frozen development nested OOF AP is **0.8889090247836890**. `ra3d reproduce` evaluates both released 2D and 3D prediction sets. The nested base arrays are included in the `baseline2d` tier and the corresponding relation/residual kernels are public.

Refit nested 2D evaluation with `python -m ra3d.baseline2d reproduce --data data --output runs/2d-nested --device cpu`. Add `--reuse-scores` only to request the published OOF/nested arrays explicitly; otherwise its base classifiers are recomputed with the required two-scene exclusions.

## Selection and training contract

The paper uses an XY-only permissive pool. Each scene's static budget equals that scene's 3D static-shortlist row count. Selection guarantees the highest-screening-score candidate for each vanishing track, then fills the remaining budget globally with deterministic frame/track ties. The final set combines the static global top 25% with structural top 20 per vanishing track. These rules matter when comparing matched computation budgets.

HGB settings and ensemble seeds match 3D. The 2D relation target uses identity-selected positives; the 3D relation target uses spatial positives. The 2D relation model uses its own 33 context plus 8 relative features, normalization and source policy. Time/x/y residual regressors correct the selected events, and projected-XY event evaluation is shared with 3D.

## Infer on your own trajectories

```bash
ra3d download --tier models --output data
python -m ra3d.baseline2d infer --tracks my-scene/tracks.parquet \
  --model data/models/baseline2d --scene-id 1 --output runs/2d-scene
ra3d review --predictions runs/2d-scene/predictions.parquet \
  --tracks my-scene/tracks.parquet --workspace runs/2d-review
```

The trajectory schema remains x/y/z, but the 2D pipeline replaces Z and Z velocity by zero before computing its depth-free candidates/features. The current command uses the paper's default units (10 µm/pixel, 4,000 frames/s). Convert custom measurements to that convention or call the feature API with explicit units and train/evaluate a corresponding model.

For matched-budget comparison, supply `--static-budget <3D shortlist row count>` explicitly. Without that argument, the command derives a gate-plus-40%-outside budget from its XY pool; that is a usable standalone policy but does not reproduce a matched 3D computation budget automatically. Full 3D inference can provide `static77.parquet` for measuring the intended budget.

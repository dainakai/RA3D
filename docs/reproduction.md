# Reproducing the paper

There are three distinct starting points: evaluate frozen predictions, refit models from frozen stage tables, or regenerate upstream measurements and candidate tables. Record which one you used.

## Evaluate released predictions

```bash
ra3d download --tier predictions --tier labels --output data
ra3d reproduce --data data --output runs/frozen-evaluation
```

This computes 3D and 2D development/terminal PR curves from the released predictions and exhaustive simulator events.

| Method / split | Expected AP |
| --- | ---: |
| 3D development nested OOF | 0.9110463791645327 |
| 3D terminal validation | 0.9171254959438848 |
| 2D development nested OOF | 0.8889090247836890 |
| 2D terminal validation | 0.8993428951464474 |

All four values were reproduced from the released Parquet exports. Evaluation uses maximum one-to-one matching within ±5 frames and XY 350 µm, without Z or particle identity in the matching gate. The separate `--protocol 3d` evaluator must not be substituted when comparing this table.

## Fit the 3D deployment model

```bash
ra3d download --tier features --tier labels --split train --output data
ra3d train --paper-root data --output models/refit --device cuda:0
```

This fits the 24-feature screen, 77-feature static classifier, five out-of-fold full-stage ensembles, relation attention, final three-model 92-feature ensemble, and four coordinate regressors. It does not read terminal validation scenes. The completed model is portable: trees use checked `skops`, attention uses `safetensors`, and normalization uses numeric NPZ. Model files have checksums, feature order and training provenance.

To score the terminal table:

```bash
ra3d download --tier features --tier labels --split validation --output data
ra3d predict --features data/features/validation/seed260745/paper92.parquet \
  --model models/refit --scene-id 260745 --output runs/refit-terminal.parquet
ra3d evaluate --predictions runs/refit-terminal.parquet \
  --truth data/labels/validation/seed260745/collision_events.parquet \
  --output runs/refit-terminal-evaluation
```

## Refit nested development evaluation

```bash
ra3d download --tier features --tier labels --output data
ra3d reproduce --data data --output runs/nested-refit --refit --device cuda:0
```

The default recomputes base OOF/nested OOF scores. An accelerated, explicitly audited alternative is:

```bash
ra3d download --tier scores --output data
ra3d reproduce --data data --output runs/nested-score-reuse \
  --refit --reuse-scores --device cuda:0
```

Score reuse is recorded in output metadata. Arrays must retain the exact published candidate row order. It is not appropriate for changed features, labels, upstream policies or splits. `ra3d train --reuse-oof data/scores` similarly requests explicit deployment-fit reuse; the ordinary command recomputes those scores.

Nested relation evaluation excludes both the outer test scene and the scene being scored when fitting its base classifier. The upstream selected candidate tables are fixed scene-wise intermediates, as in the paper. This does not constitute fully nested refitting of every upstream detector and candidate-preparation stage.

## Numerical verification of the port

| Check | Observed result |
| --- | --- |
| Real candidate fixture: 77 trajectory and 11 raw image features | All eight reference rows matched at rtol 1e-9 / atol 1e-8 |
| Five independently refitted full-stage OOF arrays | Bit-for-bit identical to the historical arrays |
| Full synthetic scene Python vs Julia smoothing | Maximum Z difference 1.38e-8 slices; maximum Z-velocity difference 1.72e-10 slices/frame |
| Final 3D model refit on the original GPU class | Terminal AP exactly 0.9171254959438848 |
| Refit every outer 3D/2D relation and correction model using explicit frozen nested base scores | All four development/terminal AP values exactly matched |
| Full 6,000-frame synthetic label generation | All 2,000 events and 2,102,484 per-frame particle records matched |
| Python hologram renderer vs original Julia images, two frames × two cameras | Maximum difference one gray level; mean absolute difference below 0.00042 gray levels |

The source reference ran attention on an RTX 4080 SUPER. Repeating attention training on an RTX 4060 Ti produced a small score/AP difference despite identical base scores; the released deployment model uses the matching RTX 4080 SUPER fit. CPU execution and other supported GPUs are valid, but floating-point reduction order can change attention weights and near-tied rankings. Original image shards and frozen predictions provide exact replay assets when byte-level comparison matters.

The reference environment uses Python 3.12, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, scikit-learn 1.7.1 and PyTorch 2.12.0+cu130. The install ranges allow portable environments; use the recorded versions and GPU conditions for the closest numerical comparison. Small CI tests verify invariant behavior without downloading the benchmark.

A fresh CPU-only wheel installation was also tested with PyTorch 2.14.0+cpu and SciPy 1.18.1: all 20 tests passed, model loading and prediction worked, and local review assets were present in the wheel. The full candidate→image-feature→inference workflow was checked on the 35-frame review excerpt. [Machine-readable numerical evidence](verification.json) records the scientific comparisons.

## Regenerate upstream stages

Download `holograms`, `tracking`, `labels` and optionally `detections`/`candidates`. The scene acquisition YAML supplies exact image reconstruction geometry. [Upstream](upstream.md) explains detector/linker installation; [simulation](simulation.md) describes the Python scene generator and optical renderer. `ra3d candidates`, `ra3d features`, and `ra3d prepare-training` implement candidate and feature regeneration from trajectories.

Do not compare a deployment-model shortlist against cross-fitted published training shortlists as though their rows must be identical. Candidate selection depends on which scenes trained the upstream screening/static model. The recipe and scene exclusions must match as well as the input trajectories.

No experimental visual-review labels are used in any of these fits. The paper's experimental review is observational validation and does not supply an exhaustive recall benchmark.

# Algorithm and frozen parameters

This page describes the implemented paper recipe. The ordered model columns live in `ra3d.config`; attention and event-policy parameters are shipped in `src/ra3d/config/paper.json`. Numerical kernels include symbol-level source provenance.

## Measurement and smoothing

HoloD3 performs angular-spectrum reconstruction, minimum-intensity projection, YOLO particle detection, learned depth estimation and diameter estimation. HoloD3-Linker links the measured particles. RA3D provides a NumPy implementation of the historical polynomial Z-GMM smoother and Savitzky–Golay smoothing/velocity estimation. The [upstream guide](upstream.md) separates detector assets from collision-model assets.

Automatic Z-GMM selection evaluates polynomial degree 1/2, regularization from 1e-8 through 1, and alpha 0/0.0001/0.01 on up to 512 length-stratified tracks with at least eight observations. The original PRESS/convergence rule selects the simplest degree within 1% of the minimum score. The EM fit uses a fixed mean prior and a broad outlier Gaussian. The Savitzky–Golay boundary extension uses odd reflection, matching the original Julia implementation. Per-run selection and convergence are recorded in the smoothing audit.

## Candidates and computational budgets

For each vanishing trajectory, candidate generation searches nearby surviving trajectories. Both tracks need at least two observations. The wide pool uses an endpoint-relative window of ±12 frames, at least one survivor post-frame, an XY/Z ellipse with 1,500/30,000 µm radii, effective-distance slack of 2,000 µm, and Z weight 0.15. Boundary and vertical-catching diagnostics are features rather than ground-truth labels.

The selected gate uses ±12 frames, at least one post-frame, an ellipse with 1,200/25,000 µm radii, and finite effective margin at most 1,200 µm. The 24-feature stage is fitted on labeled selected-gate candidates. It scores the entire wide pool; all gate members plus the top **40% outside the gate** proceed to trajectory features. Counts use ceiling, with deterministic frame/track tie rules.

The 77-feature classifier scores this shortlist. Its probability and the screening probability are blended in logit space, with weights 0.75 and 0.25, to define the structural rank. Image features are acquired for the top **25% of the static shortlist globally**. The final 92-feature scoring set is the union of that image-acquired set and the structural top **20 candidates per vanishing track**.

## Image features and missing values

At each candidate time f, image aggregation uses f−10 through f−2 inclusive. Two-plane phase retrieval uses 12 iterations; the retrieved field is mean-padded to 2,048 pixels and reconstructed at the candidate depth. A 128 × 128 crop is normalized by its 1st/99th percentiles and rounded to uint8. Dark pixels at or below 30 undergo 3 × 3 closing and 2 × 2 opening. Four-connected components smaller than six pixels are removed; particle/component association uses an 11 × 11 neighborhood.

Eleven raw aggregates and four indicator features encode common components and fragmentation patterns. Some columns intentionally overlap in meaning because the frozen trained schema preserves the historical features. Non-acquired candidates have 11 raw NaNs and four derived zeros. HGB handles those NaNs directly. The `image_complete92` extension computes the same image features for every row of the final scoring union and preserves `paper_image_enriched`; it does not alter the paper's `paper92` tables.

## Labels and classifier fitting

Candidate training compatibility requires ±5 frames, XY distance at most 350 µm, and Z distance at most 15,000 µm. Maximum matching supplies event anchors. True particle IDs are used to prefer role-consistent vanishing/surviving pairs. If an event has no identity-consistent candidate, its spatially compatible candidates provide fallback positives. Other spatial candidates for an event that does have consistent identities are excluded from classifier fitting rather than relabeled negative. Candidates with no spatial match are negative.

Positive weights equalize event contribution; positive and negative total weights are balanced and the final weights have mean one. The historical float32 event factors are retained before float64 fitting.

| HGB classifier setting | Value |
| --- | --- |
| Loss | Binary log loss |
| Learning rate / iterations | 0.045 / 350 |
| Maximum leaves / minimum samples per leaf | 31 / 20 |
| L2 regularization | 0.02 |
| Early stopping | scikit-learn 1.7.1 `auto`, internal 10% row split |
| Screening and static random state | 269723 |
| Full-feature ensemble random states | 269723, 269724, 269725 |

The three full-model probabilities are averaged arithmetically. The internal early-stopping split is row based and is not the independent scene-wise performance evaluation.

## Relation attention, event selection and correction

Relation attention groups the top 20 candidates per vanishing trajectory. Its 46 inputs contain 36 candidate/context values and 10 relative values. Features are standardized using training statistics. Empty padded entries are excluded from attention, set averaging and loss. The hidden width is 48 with four attention heads and no dropout. AdamW uses learning rate 0.0008, weight decay 0.0001, 12 epochs, batch size 192 sets and seed 270901. Inference batches contain 512 sets.

The 3D relation target is **spatial compatibility**, distinct from the identity-selected classifier target. Per-scene positive/negative weights are balanced with equal positive event contribution. The final score blends base/relation logits at 0.85/0.15 and converts to within-scene percentiles.

Candidate pairs are aggregated into temporal modes separated by a 10-frame gap, retaining at most five modes per vanishing track. Coordinate suppression uses ±2 frames, XY 250 µm and Z 7,500 µm. A lineage conflict incurs a log-odds penalty of 1. Four independent HGB regressors then correct time/x/y/z without changing selection or scores. They use 150 iterations, learning rate 0.05, 31 leaves, minimum leaf size 30, L2 0.1 and disabled early stopping. Their training positives use identity selection and event-balanced weights. Corrections are clipped to ±5 frames, ±350 µm for x/y and ±15,000 µm for z; time uses nearest-even integer rounding.

## Independent evaluation

Development scenes are seeds 260740–260744; terminal validation is 260745. The latter is absent from every fit and normalization. Relation development evaluation trains base classifiers excluding both the outer evaluation scene and the context-scoring scene. The upstream scene-wise candidate tables remain fixed: this is not nesting the entire image/candidate preparation pipeline. Experimental review labels are excluded from all paper fits.

The paper's performance evaluator uses maximum one-to-one matching within ±5 frames and XY 350 µm, with no Z or identity gate. Precision/recall are pooled over independent scenes, score ties are processed together, and AP integrates the interpolated precision envelope. [Reproduction](reproduction.md) gives the expected values and validation evidence.

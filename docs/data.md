# Selective data release

The [Hugging Face dataset](https://huggingface.co/datasets/dnakaikit/RA3D) contains independent stages of the six-scene synthetic collision benchmark and the separate HoloD3 measurement training/evaluation corpus. The package pins an immutable dataset commit in `config/release.json`. `manifest.json` records every artifact's tier, scene, split, byte count and SHA-256. It also records archive member counts and any relative extraction root. Model artifacts share this dataset repository for one reproducible download interface.

## Download only what you need

```bash
ra3d list-data
ra3d download --tier features --tier labels --split train --output data --plan
ra3d download --tier features --tier labels --split train --output data
ra3d download --tier models --output data
```

Repeat `--tier` or `--scene` to combine selections. `--scene 260745` and `--scene seed260745` are equivalent. `--split train` selects development scenes; `--split validation` selects the terminal scene. Assets that are independent of scene/split, such as models, remain included when their tier is requested. `--plan` lists exact paths and download bytes without downloading the artifacts. The small manifest itself is fetched to plan the selection.

| Tier | Contents | Typical use |
| --- | --- | --- |
| `demo` | 12 candidates, trajectories and 35 paired frames from terminal validation | Test local visual review |
| `models` | Screening/static/full ensembles, relation weights, normalization, coordinate regressors and model cards | Inference without training |
| `labels` | Complete collision events, per-frame particles, initial particles and true trajectories | Fit labels, evaluation, simulation replay |
| `detections` | Raw detections and measured 3D particles | Restart measurement/linking |
| `tracking` | Raw linked trajectories, smoothed trajectories and track-to-truth identity mapping | Regenerate candidates/features |
| `candidates` | Permissive 3D candidate pool | Inspect screening recall and budgets |
| `features` | Gate training, static77, static scores and frozen paper92 tables | Refit paper collision models |
| `image-complete` | All image features for every row of the final 92-feature scoring union | New image-acquisition experiments |
| `scores` | Full-stage OOF, nested OOF and terminal intermediate scores | Explicit accelerated reproduction and audits |
| `predictions` | Released 3D/2D development and terminal predictions | Reproduce reported evaluation |
| `baseline2d` | Depth-free gate/static/full/wide tables and reference scores | Refit the 2D counterpart |
| `holograms` | All 6,000 synchronized background-removed image pairs per scene, calibration and frame manifests | Recompute image stages |
| `holod3-models` | Four upstream measurement models and their license/source notices | Detect particles, estimate depth and diameter |
| `holod3-initializers` | Two original YOLO detector initializers and AGPL notices | Retrain upstream detectors |
| `holod3-training` | Five detector, depth and diameter bundles with original train/validation/test splits | Retrain upstream measurement models |
| `holod3-evaluation` | Twelve paired synthetic frames, calibration and particle truth | Verify the measurement pipeline |

## Layout and scientific split

```text
labels/{train,validation}/seedNNNNNN/collision_events.parquet
detections/{train,validation}/seedNNNNNN/{raw_detections,particles_3d}.parquet
tracking/{train,validation}/seedNNNNNN/{trajectory3d,smoothedtrajs,track_truth_mapping}.parquet
candidates/{train,validation}/seedNNNNNN/wide.parquet
features/{train,validation}/seedNNNNNN/{static77,static_scored,paper92}.parquet
features/train/seedNNNNNN/stage1_training.parquet
features/{train,validation}/seedNNNNNN/image_complete92.parquet
predictions/{3d,2d}/{train,validation}/seedNNNNNN.parquet
holograms/{train,validation}/seedNNNNNN/{acquisition.yaml,frames.csv,frames-*.tar}
```

Seeds 260740–260744 form the development split and seed 260745 is terminal validation. Each scene has 6,000 frames and 2,000 prescribed collisions. The initial population target is 200 particles; particles are replenished and collide during the scene, so the total number of particle identities is larger. The synthetic mixture contains 1,000 coverage events and 1,000 empirical-frequency events. It is designed for algorithm evaluation and does not estimate an experimental collision rate directly.

`labels` describe true particles; `detections` describe measured particles; `tracking` describes observed trajectories. These IDs are different namespaces. Join through `track_truth_mapping`, never by assuming equal integer IDs. Hologram filename 000001 corresponds to frame 0.

## Paper features versus completed image features

`paper92` retains the actual compute policy: only the globally selected image subset has image measurements. The other candidates have raw-image NaNs and derived indicator zeros. These values and their row order are part of the frozen experiment. Replacing them by newly computed values changes the model input distribution and is a new experiment.

`image_complete92` keeps the same candidate rows and adds the original mask as `paper_image_enriched`. Every row has all 11 raw image aggregates and the four corresponding derived indicators. `image_features_computed` identifies the extension. A computed value may be zero because no valid pre-event observations were available; check `rendered_pre_frames` and `skipped_pre_frames`. Completion is over the **final scoring union**, not every row in the much larger wide pool or static shortlist. Those larger pools and all source images/tracks are available for further computation with `ra3d features --kind image`.

Feature tables also contain candidate geometry, truth annotations, ranks and provenance fields beyond the 24/77/92 actual model inputs. The explicit feature lists define what the model consumes. Parquet preserves NaNs and column types and avoids executing serialized Python objects. Numeric CSV exports should be read with round-trip floating-point parsing when exact comparison matters.

## Hologram shards and disk use

```bash
ra3d download --tier holograms --scene 260745 --output data --extract
ra3d validate-acquisition data/holograms/validation/seed260745/acquisition.yaml
```

Each image archive contains 500 synchronized pairs. Twelve shards cover each scene. Archives use uncompressed tar because PNG data are already compressed; extraction is optional and no images are downloaded implicitly. The manifest supplies exact byte counts, which are preferable to an approximate total in this document.

Allow disk space for the Hugging Face cache, downloaded archives and extracted images. The downloader verifies every artifact before use and rejects archive traversal, links and special files. It resumes already verified files and records a download receipt. To re-extract files removed manually, remove the corresponding `*.extracted.json` marker before repeating `--extract`. Cache placement can be configured through the normal Hugging Face cache environment variables.

## Licenses and scope

Original RA3D synthetic data use CC BY 4.0. RA3D collision model artifacts use MIT, as specified in their cards. The release is the training/validation corpus for the collision experiment. Experimental visual-review decisions are not exhaustive event truth and are not mixed into training. HoloD3 detector/depth/diameter training bundles and weights have their own distribution and licensing status; see [upstream assets](upstream.md).

For direct Hub use, download paths from the manifest with `hf_hub_download(..., repo_type="dataset", revision=<pinned commit>)`. Hub dataset-viewer configurations expose selected Parquet tables; the manifest-based CLI covers all stages, archives and model files.

## HoloD3 assets

The four `holod3-*` tiers are independent of the six collision scenes. Selecting them does not fetch the full collision holograms. Their original internal training/validation/test splits remain inside each bundle; the CLI's collision `--split` filter does not subdivide them.

```bash
ra3d download --tier holod3-models --output data
ra3d download --tier holod3-training --tier holod3-evaluation --extract --output data
```

Use RA3D 0.1.1 or newer: it understands the manifest's `extract_root` field and installs upstream bundles at the layout expected by HoloD3. Follow the [upstream guide](upstream.md) to clone HoloD3 at the matching destination before downloading if you intend to run its full training ledger. No original private account credentials are needed. The project-owned upstream data use CC BY 4.0; depth/diameter checkpoints use MIT; YOLO checkpoints and initializers retain AGPL-3.0. Each selected tier includes its applicable notices.

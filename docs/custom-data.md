# Training and inference on your own data

Choose the earliest stage for which you have reliable measurements. A pretrained collision model needs trajectories plus image features. Training additionally needs independent scenes and complete event truth. Raw holograms alone are not collision training labels.

## Required inputs by starting point

| Starting point | Prepare | Next action |
| --- | --- | --- |
| Paired holograms | Synchronized grayscale images, optical geometry, registration coefficients, detector/depth/diameter weights | [HoloD3 detection](upstream.md) |
| Detected particles | HoloD3 `particles_3d.csv`, including frame filenames, positions, sizes and bounding boxes | `ra3d link` |
| Unsmoothed trajectories | Track IDs, integer frames, x/y/z, diameter | `ra3d smooth` |
| Smoothed trajectories | The table below, plus acquisition YAML or a complete image-feature bank | `ra3d infer` |
| Prepared features | The exact 92-column schema, candidate keys, geometry and selection metadata | `ra3d predict` |
| New collision training | At least three independent training scenes; complete truth; identity mapping; trajectories; images or image banks | `ra3d prepare-training`, then `ra3d train` |

## Trajectory table and units

Use CSV, CSV.gz, or Parquet. A directory of `path_<integer>.csv` files is also accepted; the filename supplies `track_id` in that case. One row represents one observation. Sort order is normalized by the reader.

| Column | Type / unit | Meaning |
| --- | --- | --- |
| `track_id` | Nonnegative integer | Observed trajectory ID, unique within the scene |
| `frame` | Integer | Image-manifest frame; normally zero based |
| `x`, `y` | Pixel coordinates | The paper uses pixel indices with the first image pixel at 1; y increases downward |
| `z` | Slice coordinate | May be fractional after smoothing; the first reconstruction slice is 1 |
| `diameter_px` | Positive pixels | Equivalent particle diameter |
| `vx`, `vy`, `vz` | Pixels/frame, pixels/frame, slices/frame | Optional; finite differences are computed when absent |
| `postOutProb` | Probability | Optional Z-GMM outlier probability |

Every `(track_id, frame)` must be unique. Required coordinates and diameters must be finite. Short or gapped tracks are permitted; at least two observations are needed for a track to participate in screening. Missing observations are not filled across arbitrary gaps by the table reader.

The paper uses 10 µm/pixel, 100 µm/slice, 4,000 frames/s, and 1,024 × 1,024 images. Supply your actual values in the acquisition manifest. Existing model thresholds are in physical units, but a new imaging system can change feature distributions; validate a transferred model on independent labeled scenes.

Collision-table coordinates follow the historical convention `x_um = x * pixel_pitch_um`, `y_um = y * pixel_pitch_um`, and `z_um = z * slice_spacing_um`. **These are geometry coordinates, not optical propagation distances.** Truth events must use the same convention. The optical reconstruction distance is separately `reconstruction_start_um + (z - 1) * slice_spacing_um`.

For physical measurements referenced to the first image pixel and a known first-slice position:

```python
from ra3d.config import Units
from ra3d.io import read_table, write_table
from ra3d.tracks import from_physical

units = Units(pixel_pitch_um=10, slice_spacing_um=100, fps=4000, image_size_px=1024)
tracks = from_physical(read_table("measurements.parquet"), units,
                       z_origin_um=64900, xy_origin_px=1)
write_table(tracks, "tracks.parquet")
```

The physical input needs `track_id, frame, x_um, y_um, z_um, diameter_um`. Convert event truth into the resulting **geometry** convention too; do not mix absolute optical Z with slice-scaled Z.

## Images and optical calibration

Copy [acquisition.yaml](../examples/acquisition.yaml) and [frames.csv](../examples/frames.csv) into your scene directory. Paths are relative to the YAML directory. The frame manifest explicitly associates each integer frame with primary and secondary images, so filenames need not follow a prescribed pattern.

```yaml
schema_version: 1
frames: frames.csv
units:
  pixel_pitch_um: 10.0
  slice_spacing_um: 100.0
  fps: 4000.0
  image_size_px: 1024
optics:
  mode: dual_phase_retrieval
  wavelength_um: 0.6328
  phase_distance_um: 21450.0
  reconstruction_start_um: 64900.0
  padding_px: 2048
  phase_iterations: 12
  calibration_coefficients: [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
```

The numerical values above describe the released synthetic scenes; replace them with measured values for your instrument. Calibration contains 12 coefficients in the source quadratic-transform order: six for source x and six for source y, using `[1, x, y, x*x, x*y, y*y]` at one-based output pixel coordinates. The sampler rounds the mapped coordinates to integer pixels and fills out-of-bounds pixels with the image mean. A text-file path can replace the list. The identity coefficients shown above are appropriate only when the images already share the required registration. The primary image is C002 and the secondary image is C001 in the paper data.

```csv
frame,primary,secondary
0,holograms/primary/000001.png,holograms/secondary/000001.png
1,holograms/primary/000002.png,holograms/secondary/000002.png
```

Images must be grayscale, square, and synchronized. The reference image loader maps 8-bit gray values to [0, 1]; use the same background-removal/intensity preparation as your upstream measurement pipeline. The release supplies the background-removed image pairs used for reconstruction. Missing image files are errors. Frames beyond the acquisition boundary are skipped by image-feature windows; an interior missing frame is an error.

`single_gabor` mode supports a single primary image with the same geometry fields. It is an alternative acquisition workflow, not the two-plane paper protocol. Omit secondary paths and calibration for that mode.

```bash
ra3d validate-acquisition my-scene/acquisition.yaml
ra3d smooth --tracks my-scene/trajectory3d.parquet \
  --output my-scene/tracks.parquet
ra3d infer --tracks my-scene/tracks.parquet \
  --acquisition my-scene/acquisition.yaml --model data/models/paper \
  --scene-id 100 --output runs/my-scene --device cuda:0
ra3d review --run runs/my-scene --device cuda:0
```

Inference writes the wide pool, static shortlist, acquired image features, full scoring table, predictions in Parquet/CSV, materialized tracks, and `run.json`. Reuse the same command after an interrupted image computation to resume its checkpoint. Use a new output directory after a completed inference.

Without images, supply `--image-bank bank.parquet`. The bank must contain the candidate keys and all 15 image features for every candidate selected for image acquisition. Its values must come from the same trajectories, calibration and image-processing recipe. A missing bank row is rejected. For an intentionally image/depth-free experiment, use the separately trained [2D method](baseline2d.md).

## Event truth and particle identity

A truth table contains **one row per physical collision event**, including events missed by the candidate generator:

| Column | Meaning |
| --- | --- |
| `collision_event_id` | Integer event ID, unique within a scene |
| `frame` | Integer event time in the trajectory time base |
| `x_um`, `y_um`, `z_um` | Finite event position in the geometry convention above |
| `parent1_id`, `parent2_id`, `child_id` | Integer true particle identities before and after coalescence |
| `sampling_component` | Optional sampling stratum; defaults to `custom` |

The identity mapping is a separate table with `frame, observed_track_id, source_particle_id`. It maps observed track segments onto actual particle identity at each frame. Each `(frame, observed_track_id)` must be unique. Use `source_particle_id = -1` for unknown identity, or omit that observation from the mapping. An empty mapping with these column headers is accepted when identity is entirely unknown; spatially compatible positives then use the documented fallback. State this limitation when reporting a custom experiment.

To build truth from manual annotation, inspect full independent sequences and record all physical events, their time/location and parent/child identities. Reviewing only high-scoring candidates does not establish that all other candidates or frames are negative. The local review export is useful evidence for annotation and active learning, but it is **not automatically converted into exhaustive training truth**.

## Prepare independent scenes and train

Split by independent acquisition or simulation seed before preparing candidates. Neighboring frames from the same recording are not independent scenes. Keep terminal validation scenes out of feature-policy fitting, classifiers, normalization, relation attention and residual regressors.

Copy [training-project.yaml](../examples/training-project.yaml). Each scene supplies paths relative to the project file:

```yaml
schema_version: 1
scenes:
  - id: 101
    split: train
    exhaustive_truth: true
    tracks: scene101/tracks.parquet
    acquisition: scene101/acquisition.yaml
    truth: scene101/events.csv
    mapping: scene101/identity.csv
  # Add at least two more independent train scenes.
  # Add one or more independent validation scenes with split: validation.
```

If using an image bank, replace `acquisition` with `image_bank` and supply the `units` mapping explicitly. `exhaustive_truth: true` is an assertion about your annotation coverage; set it only after that coverage is established.

```bash
ra3d prepare-training --project my-data/project.yaml \
  --output data/prepared-custom --device cuda:0
ra3d train --prepared data/prepared-custom \
  --output models/custom --device cuda:0
ra3d predict --features data/prepared-custom/validation/scene201/full.parquet \
  --model models/custom --scene-id 201 --output runs/custom-validation.parquet
ra3d evaluate --predictions runs/custom-validation.parquet \
  --truth my-data/scene201/events.csv --output runs/custom-validation-metrics
```

Preparation scores each training scene with upstream models trained on the other training scenes. Validation preparation uses all training scenes. The final fit reads only `train` entries from `prepared.json`; at least three train scenes are needed for scene-separated relation training. Preparation and model fitting can require substantial RAM because multiple scene tables are held together. The reference deployment fit uses approximately 1.26 million full-feature training rows.

The default fit recomputes out-of-fold base scores. Completed classifier stages are checkpointed, with input hashes and parameters checked on restart. `--reuse-oof` is an explicit advanced option for audited scores with identical row order and scene exclusions. Do not replace missing features with zero or train the paper model on the completed-image extension without defining a new experimental recipe.

## Prediction and review identity

`score` is a within-scene ranking score, not a calibrated collision probability. `candidate_frame` and `x_um/y_um/z_um` are corrected event estimates. `candidate_frame_raw` and `*_um_raw` preserve the selected source candidate. `candidate_id` combines scene, vanished track, survivor track and the **raw** frame so that review labels remain stable after coordinate correction.

Use the `projected-xy` evaluation protocol for comparison with the paper. The optional `3d` protocol adds a 15,000 µm Z gate and answers a different question. Report annotation coverage and the chosen protocol with every metric.

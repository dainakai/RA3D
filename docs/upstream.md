# HoloD3 detection and trajectory linking

RA3D exposes explicit adapters to the public Python measurement and linking packages. The collision algorithms also work with equivalent measurements from another system, using the [trajectory schema](custom-data.md).

| Component | Source revision |
| --- | --- |
| [HoloD3](https://github.com/dainakai/HoloD3) | `bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50` |
| [HoloD3-Linker](https://github.com/dainakai/HoloD3-Linker) | `fbe935a5d7fcd6b0c04ad0b2b2b9786c23f08cdf` |

The HoloD3 adapter invokes its `portable-torch` pipeline. It does not require Julia or TensorRT. Keep a source checkout because HoloD3 resolves packaged acquisition resources relative to its repository. Install HoloD3 and RA3D in the same environment for the `ra3d detect` wrapper, or run HoloD3 in its own environment and pass the resulting files to RA3D.

```bash
git clone https://github.com/dainakai/HoloD3.git data/upstream/holod3
git -C data/upstream/holod3 checkout bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50
python -m pip install -e data/upstream/holod3
python -m pip install -e ".[tracking]"
```

For HoloD3's full training environment, use its own frozen `uv.lock` and [training guide](https://github.com/dainakai/HoloD3/blob/bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50/docs/training.md). The optional `tracking` extra installs the pinned Python linker.

## Asset availability

RA3D's `models` tier contains the **collision** models. Starting with RA3D 0.1.1, the separate HoloD3 training/evaluation bundles, four production weights and two detector initializers are also publicly available through the [RA3D dataset](https://huggingface.co/datasets/dnakaikit/RA3D). No login or access request is required. The mirror preserves the original bytes and SHA-256 values of the following pinned source artifacts:

| Upstream repository | Pinned revision | Contents |
| --- | --- | --- |
| `dnakaikit/HoloD3-datasets` | `8cee42b5290592e7cd36181edbe96d0a8bda57aa` | Six detector/depth/diameter/evaluation bundles, 518,048,228 compressed bytes |
| `dnakaikit/HoloD3-models` | `520e56fca2e6b576bba600a3b64dcb6ce59d4747` | Four production models and two detector initializers, approximately 324 MB |

Choose only the part you need:

| Tier | Assets | Compressed/checkpoint bytes, excluding small notices |
| --- | --- | ---: |
| `holod3-models` | Detector, primary depth, fallback depth and diameter production weights | 217,812,731 |
| `holod3-initializers` | Detector architecture and packaged baseline for retraining | 106,565,322 |
| `holod3-training` | Five training bundles, including their original validation/test splits | 493,860,706 |
| `holod3-evaluation` | Twelve paired synthetic benchmark frames, calibration and particle truth | 24,187,522 |

From the RA3D checkout, download the production weights into the HoloD3 clone created above:

```bash
ra3d download --tier holod3-models --output data --plan
ra3d download --tier holod3-models --output data
```

Checkpoints are installed below `data/upstream/holod3/models/production/`. Dataset archives retain their original bundle-root layout; `--extract` installs them below `data/upstream/holod3/data/downloaded/`, as required by HoloD3's training ledger. License notices and source provenance accompany each tier under `data/upstream/holod3/ra3d-release/`. The `--split` and `--scene` filters refer to RA3D collision scenes; each HoloD3 training bundle contains its own original splits and is independent of those filters.

The approved project-owned HoloD3 data use CC BY 4.0, and the primary-depth, fallback-depth and diameter checkpoints use MIT. The YOLO detector and both detector initializers retain **Ultralytics AGPL-3.0** terms, with the license text and corresponding source links supplied alongside them. RA3D's MIT license does not replace those terms. Historical HoloD3 notices are retained for provenance; their previous private-access wording predates this explicitly authorized publication. See [third-party notices](../THIRD_PARTY_NOTICES.md) and [Ultralytics licensing](https://www.ultralytics.com/license).

HoloD3's source manifests list filenames, sizes and checksums. Its training ledger documents experimental/mixed YOLO data, primary/fallback depth crops, diameter crops, validation splits, initializers and checkpoint selection. Those manifests are the authority for upstream model reproduction; RA3D's six 6,000-frame scenes are the collision corpus, not a replacement for those upstream crop-training datasets.

## Prepare upstream retraining without an account

With the HoloD3 clone in place, run the following from the RA3D checkout:

```bash
ra3d download --tier holod3-models --tier holod3-initializers \
  --tier holod3-training --tier holod3-evaluation --extract --output data
cd data/upstream/holod3
uv sync --frozen --extra gpu --extra train
uv run holod3 verify
uv run holod3 reproduce --stage check
uv run holod3 reproduce --stage all --dry-run
```

After inspecting the dry run, use `uv run holod3 reproduce --stage all` to train. To feed a newly trained detector base into production fine-tuning, add `--yolo-baseline-source reproduced`; the default deliberately uses the published baseline initializer. The [upstream training guide](https://github.com/dainakai/HoloD3/blob/bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50/docs/training.md) records hyperparameters, checkpoint selection and reproducibility limits. Its older preparation instructions mention authentication; the RA3D download commands above replace those fetch/login steps. Downloads include fixed upstream artifacts, not newly retrained HoloD3 models.

The release verification resolved every detector image/label, 104,172 primary-depth crops, 30,600 fallback-depth crops, 14,080 diameter crops and all 12 paired benchmark frames through HoloD3's own input checker. All six checkpoint hashes matched. A separate one-frame GPU inference check produced 485 measured particles using the released weights; this is an execution check, not an accuracy estimate. [Machine-readable evidence](upstream-verification.json) records the checks and their scope.

## Detect particles from your images

HoloD3 uses a **directory-based acquisition YAML**. RA3D uses a **frame-CSV acquisition YAML**. They describe the same imaging geometry but are different schemas; do not pass one unchanged to the other. HoloD3 pairs files by stem. RA3D pairs files by explicit integer frame.

Copy [holod3-acquisition.yaml](../examples/holod3-acquisition.yaml) and set paths and measured optical values. Omit `frames.minip` to generate minimum-intensity projections. Supply your detector/depth/diameter model paths explicitly:

```bash
ra3d detect --holod3-root data/upstream/holod3 \
  --acquisition my-scene/holod3-acquisition.yaml --output runs/measurement \
  --yolo-weights data/upstream/holod3/models/production/detector.pt \
  --depth-primary-weights data/upstream/holod3/models/production/depth-primary.pt \
  --depth-fallback-weights data/upstream/holod3/models/production/depth-fallback.pt \
  --diameter-weights data/upstream/holod3/models/production/diameter.pt --device cuda:0 --plan
```

Remove `--plan` to run; `--limit 0` processes all frames. A nonzero `--limit` is useful for checking a new instrument's configuration. HoloD3 writes `particles_3d.csv` and measurement provenance. Its transformed image coordinates must match the images used later for RA3D features; persist any crop/flip/registration transforms consistently.

The paper's collision image features use 12 phase iterations and 2,048-pixel padding. Upstream detector projections have their own reconstruction settings; do not silently replace collision-feature settings with the upstream fast-projection preset.

## Link and smooth

```bash
ra3d link --particles runs/measurement/particles_3d.csv \
  --output runs/linking --pixel-pitch-um 10 --slice-spacing-um 100
ra3d smooth --tracks runs/linking/trajectory3d.parquet \
  --output runs/smoothed.parquet
```

The linker adapter also accepts the public `particles_3d.parquet` download. Its input requires `frame, file, conf, xc, yc, w, h, slice, diameter_px`; the HoloD3 table supplies additional segmentation, physical coordinate and final-diameter fields that the adapter uses when present. The `file` column is a portable image basename in the RA3D release.

The frozen linker uses fixed-lag linking, segmentation XY, bounding-box area and Z as a tie breaker (`z_weight=0.10`, margin 0.04, penalty 0.10, cap 1.0). Exported tracks retain the original slice coordinate. `trajectory3d.parquet` uses the RA3D table schema; `ra3d_link.json` records source and units. The smoother is entirely Python and writes a selection/convergence audit next to its output.

Create RA3D's `frames.csv` with the example helper if filenames are numeric:

```bash
python examples/make_frame_manifest.py --root my-scene \
  --primary holograms/primary --secondary holograms/secondary \
  --filename-offset 1 --output frames.csv
```

Here filename 000001 maps to frame 0. For other naming conventions, write the three-column CSV explicitly. Then use the [RA3D acquisition example](../examples/acquisition.yaml), validate it, and run collision inference.

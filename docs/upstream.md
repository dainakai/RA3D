# HoloD3 detection and trajectory linking

RA3D exposes explicit adapters to the public Python measurement and linking packages. The collision algorithms also work with equivalent measurements from another system, using the [trajectory schema](custom-data.md).

| Component | Source revision |
| --- | --- |
| [HoloD3](https://github.com/dainakai/HoloD3) | `bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50` |
| [HoloD3-Linker](https://github.com/dainakai/HoloD3-Linker) | `fbe935a5d7fcd6b0c04ad0b2b2b9786c23f08cdf` |

The HoloD3 adapter invokes its `portable-torch` pipeline. It does not require Julia or TensorRT. Keep a source checkout because HoloD3 resolves packaged acquisition resources relative to its repository. Install HoloD3 and RA3D in the same environment for the `ra3d detect` wrapper, or run HoloD3 in its own environment and pass the resulting files to RA3D.

```bash
git clone https://github.com/dainakai/HoloD3.git ../HoloD3
git -C ../HoloD3 checkout bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50
python -m pip install -e ../HoloD3
python -m pip install -e ".[tracking]"
```

For HoloD3's full training environment, use its own frozen `uv.lock` and [training guide](https://github.com/dainakai/HoloD3/blob/bdd7cf0d1ce57a86e7e9205b493ecd5fdc04bc50/docs/training.md). The optional `tracking` extra installs the pinned Python linker.

## Asset availability

RA3D's public `models` tier contains the **collision** models. HoloD3's detector, depth and diameter weights are separate assets. At the initial RA3D preparation date, the existing HoloD3 asset repositories are private:

| Upstream repository | Pinned revision | Contents |
| --- | --- | --- |
| `dnakaikit/HoloD3-datasets` | `8cee42b5290592e7cd36181edbe96d0a8bda57aa` | Six detector/depth/diameter/evaluation bundles, 518,048,228 compressed bytes |
| `dnakaikit/HoloD3-models` | `520e56fca2e6b576bba600a3b64dcb6ce59d4747` | Four production models and two detector initializers, approximately 324 MB |

They are not included under RA3D's data or software license. Access to those private upstream repositories is required for their existing `fetch-data`/`fetch-models` commands. You can use your own trained weights with the public code. The public RA3D detections, trajectories, features and collision models allow all downstream collision training and evaluation without these private assets. This access distinction is explicit so an anonymous user is not directed to a download that silently requires owner credentials.

HoloD3's source manifests list filenames, sizes and checksums. Its training ledger documents experimental/mixed YOLO data, primary/fallback depth crops, diameter crops, validation splits, initializers and checkpoint selection. Those manifests are the authority for upstream model reproduction; RA3D's six 6,000-frame scenes are the collision corpus, not a replacement for those upstream crop-training datasets.

## Detect particles from your images

HoloD3 uses a **directory-based acquisition YAML**. RA3D uses a **frame-CSV acquisition YAML**. They describe the same imaging geometry but are different schemas; do not pass one unchanged to the other. HoloD3 pairs files by stem. RA3D pairs files by explicit integer frame.

Copy [holod3-acquisition.yaml](../examples/holod3-acquisition.yaml) and set paths and measured optical values. Omit `frames.minip` to generate minimum-intensity projections. Supply your detector/depth/diameter model paths explicitly:

```bash
ra3d detect --holod3-root ../HoloD3 \
  --acquisition my-scene/holod3-acquisition.yaml --output runs/measurement \
  --yolo-weights my-models/detector.pt \
  --depth-primary-weights my-models/depth-primary.pt \
  --depth-fallback-weights my-models/depth-fallback.pt \
  --diameter-weights my-models/diameter.pt --device cuda:0 --plan
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

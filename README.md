# RA3D

**Reproduce, train, and review three-dimensional particle collisions from digital holography.**

RA3D packages the algorithms used in *Holographic Droplet Collision Measurement*. It includes collision candidate generation, 24/77/92-feature classification, relation attention, event selection, coordinate correction, a depth-free 2D counterpart, Python trajectory smoothing, synthetic scene generation, and a local visual review application. Detection, depth estimation, diameter estimation, and trajectory linking integrate with [HoloD3](https://github.com/dainakai/HoloD3) and [HoloD3-Linker](https://github.com/dainakai/HoloD3-Linker).

All computation is Python. CPU execution is supported; CUDA accelerates holographic reconstruction and attention training. The repository contains code and small numerical fixtures. Large data and model assets are downloaded selectively from the [RA3D dataset on Hugging Face](https://huggingface.co/datasets/dnakaikit/RA3D).

## Install

Use **Python 3.12** in a fresh environment. Scikit-learn is pinned because its trained tree representation depends on its version.

```bash
git clone https://github.com/dainakai/RA3D.git
cd RA3D
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
ra3d --help
```

For CUDA, install a PyTorch build appropriate for your driver before installing RA3D. Use `--device cuda:0` on reconstruction, inference, training, and review commands. `--device cpu` is the default. Install trajectory linking with `python -m pip install -e ".[tracking]"`; HoloD3 uses its own documented source installation.

## Start with the small review example

```bash
ra3d download --tier demo --output data --extract
ra3d review --predictions data/demo/predictions.parquet \
  --tracks data/demo/tracks.parquet \
  --acquisition data/demo/acquisition.yaml \
  --workspace runs/demo-review
```

Open **http://127.0.0.1:8765**. Rotate the 3D trajectories, play frames, build a reconstructed event movie, and label candidates as collision, noncollision, or ambiguous. Notes and revision history are stored in local SQLite; export labels directly from the page. No cloud account, external JavaScript, or review server is required. See [local review](docs/review.md).

## Choose data by purpose

```bash
ra3d list-data
ra3d download --tier features --tier labels --split train --output data --plan
ra3d download --tier features --tier labels --split train --output data
ra3d download --tier models --output data
```

| Your task | Tiers to download |
| --- | --- |
| Refit the collision model | `features`, `labels` |
| Evaluate the paper's frozen results | `predictions`, `labels` |
| Predict from precomputed paper features | `features`, `models` |
| Recompute candidates or trajectory features | `tracking` |
| Start from detected particles | `detections` |
| Use completed image features in a new experiment | `image-complete` |
| Reconstruct images or regenerate image features | `holograms`, `tracking` |
| Reproduce the depth-free counterpart | `baseline2d`, `labels` |
| Run upstream particle detection, depth and diameter estimation | `holod3-models` |
| Retrain and check upstream measurement models | `holod3-training`, `holod3-evaluation`, `holod3-models`, `holod3-initializers` |

Tiers, scenes, splits, and exact byte sizes are recorded in a checksum manifest. Holograms are never downloaded implicitly. To obtain one scene's paired images:

```bash
ra3d download --tier holograms --scene 260745 --extract --output data
```

See [data layout and download controls](docs/data.md), including the distinction between `paper92` and `image_complete92`.

HoloD3 training data and measurement checkpoints are publicly downloadable without an account. Its four optional tiers add about 842 MB before extraction, with separate license notices for YOLO assets. The [upstream guide](docs/upstream.md) gives the clone, download, verification and retraining commands.

## Reproduce the paper

```bash
ra3d download --tier predictions --tier labels --output data
ra3d reproduce --data data --output runs/paper-evaluation
```

The evaluator uses maximum one-to-one event matching under the paper's projected-XY protocol: ±5 frames and 350 µm in XY, without a Z gate.

| Method | Development nested OOF AP | Terminal validation AP |
| --- | ---: | ---: |
| RA3D 3D | 0.9110463792 | 0.9171254959 |
| Depth-free 2D | 0.8889090248 | 0.8993428951 |

To train the complete 3D deployment model from the staged data:

```bash
ra3d download --tier features --tier labels --split train --output data
ra3d train --paper-root data --output models/refitted --device cuda:0
```

Completed classifiers and training stages are checkpointed. Reissue the same command after interruption. To recompute nested evaluation as well, download both splits and run `ra3d reproduce --data data --output runs/refit --refit --device cuda:0`. This refits from the published scene-wise candidate tables. [Reproduction documentation](docs/reproduction.md) explains upstream regeneration, split boundaries, explicit score reuse, and numerical tolerances.

## Use your own data

Start at the stage you already have:

```text
Synchronized holograms + optical calibration
    → HoloD3 detections / depth / diameter
    → HoloD3-Linker trajectories
    → Python Z-GMM / Savitzky–Golay smoothing
    → Collision candidates → 77 trajectory features
    → Selected image reconstructions → 92 features
    → Relation attention → Event selection → Coordinate correction
    → Local visual review
```

Given smoothed trajectories and synchronized image manifests:

```bash
ra3d infer --tracks my-scene/tracks.parquet \
  --acquisition my-scene/acquisition.yaml \
  --model data/models/paper --scene-id 1 \
  --output runs/my-scene --device cuda:0
ra3d review --run runs/my-scene --device cuda:0
```

For existing feature tables, use `ra3d predict --features ... --model ... --output ...`. For training, prepare independent labeled scenes in a project YAML, run `ra3d prepare-training`, then `ra3d train --prepared ...`. The [custom-data guide](docs/custom-data.md) defines every required input, coordinate convention, truth annotation, split, and command. Sparse visual reviews are not automatically treated as exhaustive event truth.

## Documentation

- [Algorithm and frozen parameters](docs/algorithm.md)
- [Feature catalog](docs/features.md)
- [Selective datasets](docs/data.md)
- [Training and inference on custom data](docs/custom-data.md)
- [Paper reproduction and validation evidence](docs/reproduction.md)
- [HoloD3 and tracking integration](docs/upstream.md)
- [Local visual review](docs/review.md)
- [Synthetic scenes and hologram generation](docs/simulation.md)
- [Depth-free 2D counterpart](docs/baseline2d.md)
- [Physical consistency and gravitational collision rates](docs/physics.md)

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
```

Tests cover real trajectory-feature reference values, event matching, image-checkpoint recovery, smoothing, safe model round trips, selective data extraction, and review persistence. Larger numerical comparisons are documented separately from the small CI suite.

## Attribution and licenses

RA3D-specific software and collision model artifacts are released under the **MIT License**. Original released synthetic research data use **CC BY 4.0**. Upstream code, datasets, and model checkpoints retain their applicable licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The HoloD3 asset access status is documented explicitly in the [upstream guide](docs/upstream.md).

Please cite the paper, repository version, and dataset revision used in your work. [CITATION.cff](CITATION.cff) contains repository citation metadata. Numerical source provenance is recorded in the package, including source-file hashes and original symbols.

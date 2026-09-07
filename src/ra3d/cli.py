"""Command-line entry point for the RA3D research and review workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def emit(value):
    from .io import json_value

    print(json.dumps(json_value(value), allow_nan=False), flush=True)


def parser():
    root = argparse.ArgumentParser(
        prog="ra3d", description="Reproduce, train, infer, and review 3D particle collisions."
    )
    root.add_argument("--version", action="version", version="RA3D 0.1.0")
    sub = root.add_subparsers(dest="command", required=True)

    def command(name, description):
        return sub.add_parser(name, help=description, description=description)

    def runtime(p):
        p.add_argument("--device", default="cpu", help="cpu or cuda:N; CPU is the portable default")
        p.add_argument("--cpu-threads", type=int, default=4)

    def trajectory(p):
        p.add_argument(
            "--tracks", required=True, help="Trajectory CSV/Parquet or directory of path_*.csv files"
        )
        p.add_argument("--acquisition", help="RA3D acquisition.yaml")

    p = command("download", "Selectively download verified data and model artifacts")
    p.add_argument(
        "--tier",
        action="append",
        required=True,
        help="Repeat for additional tiers; no holograms are downloaded implicitly",
    )
    p.add_argument("--scene", action="append")
    p.add_argument("--split", choices=["train", "validation"])
    p.add_argument("--output", default="data")
    p.add_argument("--repo", default="dnakaikit/RA3D")
    p.add_argument("--revision")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--plan", action="store_true")
    p = command("list-data", "List dataset tiers and their download sizes")
    p.add_argument("--repo", default="dnakaikit/RA3D")
    p.add_argument("--revision")
    p.add_argument("--manifest")
    p = command("validate-acquisition", "Validate image synchronization and optical geometry")
    p.add_argument("acquisition")
    p = command("detect", "Run the public HoloD3 Python detector/depth/diameter pipeline")
    p.add_argument("--holod3-root", required=True)
    p.add_argument("--acquisition", required=True, help="HoloD3 acquisition YAML")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--plan", action="store_true")
    for role in ["yolo", "depth-primary", "depth-fallback", "diameter"]:
        p.add_argument("--" + role + "-weights")
    runtime(p)
    p = command("link", "Link detected particles into trajectories with HoloD3-Linker")
    p.add_argument("--particles", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pixel-pitch-um", type=float, default=10.0)
    p.add_argument("--slice-spacing-um", type=float, default=100.0)
    p = command("smooth", "Apply Python Z-GMM and Savitzky–Golay trajectory smoothing")
    p.add_argument("--tracks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--selection", choices=["auto", "fixed"], default="auto")
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--lambda-other", type=float, default=1e-4)
    p.add_argument("--em-max-iter", type=int, default=100)
    p = command("candidates", "Generate the permissive collision candidate pool")
    trajectory(p)
    p.add_argument("--output", required=True)
    p = command("features", "Compute trajectory or reconstructed-image candidate features")
    trajectory(p)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["static", "image"], required=True)
    runtime(p)
    p = command("infer", "Run the complete collision cascade on your trajectories")
    trajectory(p)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--image-bank")
    p.add_argument("--scene-id", type=int, default=0)
    runtime(p)
    p = command("predict", "Score precomputed paper92 feature tables")
    p.add_argument("--features", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scene-id", type=int, default=0)
    runtime(p)
    p = command("prepare-training", "Prepare scene-separated custom training and validation tables")
    p.add_argument("--project", required=True)
    p.add_argument("--output", required=True)
    runtime(p)
    p = command("train", "Fit a complete collision model, resuming completed stages")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--paper-root")
    source.add_argument("--prepared")
    p.add_argument("--output", required=True)
    p.add_argument("--reuse-oof", help="Explicit directory of audited OOF scores; default recomputes")
    runtime(p)
    p = command("evaluate", "Evaluate corrected events against exhaustive event truth")
    p.add_argument("--predictions", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--score-column", default="score")
    p.add_argument("--protocol", choices=["projected-xy", "3d"], default="projected-xy")
    p = command("reproduce", "Evaluate released results or refit the paper models")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--refit", action="store_true")
    p.add_argument(
        "--reuse-scores", action="store_true", help="Explicitly reuse released OOF scores in refitting"
    )
    runtime(p)
    p = command("review", "Start the offline visual review application at 127.0.0.1")
    p.add_argument("--run", help="Inference output directory; supplies predictions, tracks, and acquisition")
    p.add_argument("--predictions")
    p.add_argument("--tracks")
    p.add_argument("--workspace")
    p.add_argument("--acquisition")
    p.add_argument("--score-column", default="score")
    p.add_argument("--port", type=int, default=8765)
    runtime(p)
    return root


def run(args):
    from .io import read_table, write_table, write_json

    if hasattr(args, "cpu_threads") and args.cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    name = args.command
    if name == "download":
        from .datasets import download

        return download(
            args.output,
            tiers=args.tier,
            scenes=args.scene,
            split=args.split,
            repo_id=args.repo,
            revision=args.revision,
            extract=args.extract,
            dry_run=args.plan,
            progress=emit,
        )
    if name == "list-data":
        from .datasets import load_manifest

        manifest = load_manifest(repo_id=args.repo, revision=args.revision, local=args.manifest)
        tiers = {}
        for item in manifest["files"]:
            group = tiers.setdefault(item["tier"], {"files": 0, "bytes": 0})
            group["files"] += 1
            group["bytes"] += item["bytes"]
        return tiers
    if name == "validate-acquisition":
        from .acquisition import Acquisition

        return Acquisition.load(args.acquisition).validate()
    if name == "detect":
        from .upstream import detect

        weights = {
            role: getattr(args, role.replace("-", "_") + "_weights")
            for role in ["yolo", "depth-primary", "depth-fallback", "diameter"]
        }
        detect(
            args.acquisition,
            args.output,
            holod3_root=args.holod3_root,
            device=args.device,
            weights={k: v for k, v in weights.items() if v},
            limit=args.limit,
            dry_run=args.plan,
        )
        return {"output": args.output}
    if name == "link":
        from .upstream import link

        result = link(
            args.particles,
            args.output,
            pixel_pitch_um=args.pixel_pitch_um,
            slice_spacing_um=args.slice_spacing_um,
        )
        return {"tracks": str(result.trajectory_dir)}
    if name == "smooth":
        from .smoothing import smooth_tracks, SmoothingOptions
        from .tracks import read_tracks

        result, audit = smooth_tracks(
            read_tracks(args.tracks),
            SmoothingOptions(
                selection=args.selection,
                degree=args.degree,
                alpha=args.alpha,
                lambda_other=args.lambda_other,
                em_max_iter=args.em_max_iter,
            ),
            emit,
        )
        write_table(result, args.output)
        write_json(audit, Path(args.output).with_suffix(".smoothing.json"))
        return audit["fit"]
    if name in {"candidates", "features"}:
        from .tracks import read_tracks, materialize_tracks
        from .features import generate_candidates, static_features, image_features
        from .acquisition import Acquisition, Reconstructor
        from .config import Units

        tracks = read_tracks(args.tracks)
        acq = Acquisition.load(args.acquisition) if args.acquisition else None
        if name == "candidates":
            result = generate_candidates(tracks, acq.units if acq else Units())
        else:
            candidates = read_table(args.candidates)
            if args.kind == "static":
                track_dir = materialize_tracks(tracks, Path(args.output).parent / "feature_tracks")
                result = static_features(candidates, tracks, track_dir, acq.units if acq else Units(), emit)
            else:
                if acq is None:
                    raise ValueError("Image features require --acquisition")
                result = image_features(
                    candidates,
                    tracks,
                    Reconstructor(acq, args.device),
                    progress=emit,
                    checkpoint_path=Path(args.output).with_suffix(".checkpoint.npz"),
                )
        write_table(result, args.output)
        return {"rows": len(result), "output": args.output}
    if name == "infer":
        from .workflow import infer

        result = infer(
            args.tracks,
            args.model,
            args.output,
            acquisition=args.acquisition,
            image_bank=args.image_bank,
            scene_id=args.scene_id,
            device=args.device,
            cpu_threads=args.cpu_threads,
            progress=emit,
        )
        return {"events": len(result), "output": args.output}
    if name == "predict":
        from .models import load_model
        from .cascade import predict_full
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=args.cpu_threads):
            predictions, audit = predict_full(
                read_table(args.features), load_model(args.model, device=args.device), scene_id=args.scene_id
            )
        write_table(predictions, args.output)
        write_json(audit, Path(args.output).with_suffix(".audit.json"))
        return {"events": len(predictions), "output": args.output}
    if name == "prepare-training":
        from .workflow import prepare_training

        return prepare_training(
            args.project, args.output, device=args.device, cpu_threads=args.cpu_threads, progress=emit
        )
    if name == "train":
        from .training import load_paper_training, fit_staged
        from .workflow import load_prepared

        stages, manifest = (
            load_paper_training(args.paper_root) if args.paper_root else load_prepared(args.prepared)
        )
        _, audit = fit_staged(
            stages,
            args.output,
            device=args.device,
            cpu_threads=args.cpu_threads,
            progress=emit,
            reuse_oof=args.reuse_oof,
            source_manifest=manifest,
            verify_paper=bool(args.paper_root),
        )
        return audit
    if name == "evaluate":
        from .evaluation import event_curve

        curve, metrics = event_curve(
            {0: read_table(args.predictions)},
            {0: read_table(args.truth)},
            score_column=args.score_column,
            projected_xy=args.protocol == "projected-xy",
        )
        output = Path(args.output)
        write_table(curve, output / "pr.parquet")
        write_json(metrics, output / "metrics.json")
        return metrics
    if name == "reproduce":
        from .reproduction import refit_paper, evaluate_published

        if args.reuse_scores and not args.refit:
            raise ValueError("--reuse-scores requires --refit")
        return (
            refit_paper(
                args.data,
                args.output,
                device=args.device,
                cpu_threads=args.cpu_threads,
                reuse_scores=args.reuse_scores,
                progress=emit,
            )
            if args.refit
            else evaluate_published(args.data, args.output)
        )
    if name == "review":
        from .review import create_app

        if args.run:
            root = Path(args.run)
            args.predictions = args.predictions or str(root / "predictions.parquet")
            args.tracks = args.tracks or str(root / "tracks")
            args.workspace = args.workspace or str(root / "review")
            if not args.acquisition and (root / "run.json").exists():
                args.acquisition = json.loads((root / "run.json").read_text()).get("acquisition")
        if not all([args.predictions, args.tracks, args.workspace]):
            raise ValueError("Supply --run or --predictions, --tracks and --workspace")
        if not 1024 <= args.port <= 65535:
            raise ValueError("Review port must be in 1024--65535")
        app = create_app(
            args.predictions,
            args.tracks,
            args.workspace,
            acquisition=args.acquisition,
            device=args.device,
            score_column=args.score_column,
        )
        print(f"RA3D review: http://127.0.0.1:{args.port}", flush=True)
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)
        return None
    raise ValueError(f"Unsupported command: {name}")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
        if result is not None:
            emit(result)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"RA3D: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

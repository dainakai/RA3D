"""Select, download, verify, and safely unpack immutable dataset artifacts."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from huggingface_hub import hf_hub_download
import zstandard

from .io import contained_path, sha256, write_json


DEFAULT_REPOSITORY = "dnakaikit/RA3D"


def published_revision():
    resource = files("ra3d").joinpath("config/release.json")
    if not resource.is_file():
        raise ValueError("This checkout has no published data revision; supply --revision explicitly")
    return json.loads(resource.read_text())["revision"]


def load_manifest(*, repo_id=DEFAULT_REPOSITORY, revision=None, local=None):
    if local:
        source = Path(local)
    else:
        source = Path(
            hf_hub_download(
                repo_id,
                filename="manifest.json",
                repo_type="dataset",
                revision=revision or published_revision(),
                token=False,
            )
        )
    manifest = json.loads(source.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported dataset manifest")
    paths = set()
    for item in manifest["files"]:
        path = item["path"]
        contained_path(Path.cwd(), path)
        if path in paths:
            raise ValueError(f"Duplicate dataset artifact: {path}")
        paths.add(path)
        if not isinstance(item["bytes"], int) or item["bytes"] < 0 or len(item["sha256"]) != 64:
            raise ValueError(f"Invalid checksum metadata for {path}")
    return manifest


def select_files(manifest, *, tiers, scenes=None, split=None):
    requested = set(tiers)
    available = {item["tier"] for item in manifest["files"]}
    if requested - available:
        raise ValueError(f"Unknown tiers: {sorted(requested - available)}; available: {sorted(available)}")
    scene_names = {str(s) if str(s).startswith(("seed", "scene")) else f"seed{s}" for s in scenes or []}
    known = {item.get("scene") for item in manifest["files"] if item.get("scene")}
    if scene_names - known:
        raise ValueError(f"Unknown scenes: {sorted(scene_names - known)}")
    selected = [
        item
        for item in manifest["files"]
        if item["tier"] in requested
        and (not scene_names or item.get("scene") is None or item.get("scene") in scene_names)
        and (split is None or item.get("split") is None or item.get("split") == split)
    ]
    if not selected:
        raise ValueError("No dataset files match this selection")
    return selected


def extract_archive(source, root, *, expected_members=None):
    """Reject path escapes, links, special files, and duplicate archive members."""
    source = Path(source)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ra3d-extract-", dir=root) as staging:
        seen = set()
        with source.open("rb") as raw:
            stream = zstandard.ZstdDecompressor().stream_reader(raw) if source.name.endswith(".zst") else raw
            try:
                with tarfile.open(fileobj=stream, mode="r|") as archive:
                    for member in archive:
                        if not (member.isfile() or member.isdir()):
                            raise ValueError(f"Archive contains a link or special file: {member.name}")
                        target = contained_path(staging, member.name)
                        if member.name in seen:
                            raise ValueError(f"Duplicate archive member: {member.name}")
                        seen.add(member.name)
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as reader, target.open("wb") as writer:
                            shutil.copyfileobj(reader, writer)
                        if target.stat().st_size != member.size:
                            raise ValueError("Truncated archive member")
            finally:
                if stream is not raw:
                    stream.close()
        if expected_members is not None and len(seen) != expected_members:
            raise ValueError("Archive member count differs from manifest")
        # Validate all destination paths before moving any file (including pre-existing symlinks).
        files_to_move = [p for p in Path(staging).rglob("*") if p.is_file()]
        destinations = [contained_path(root, str(p.relative_to(staging))) for p in files_to_move]
        for source_path, dest in zip(files_to_move, destinations, strict=True):
            dest.parent.mkdir(parents=True, exist_ok=True)
            source_path.replace(dest)
    return len(seen)


def download(
    output,
    *,
    tiers,
    scenes=None,
    split=None,
    repo_id=DEFAULT_REPOSITORY,
    revision=None,
    extract=False,
    dry_run=False,
    progress=None,
):
    revision = revision or published_revision()
    manifest = load_manifest(repo_id=repo_id, revision=revision)
    selected = select_files(manifest, tiers=tiers, scenes=scenes, split=split)
    report = {
        "repo_id": repo_id,
        "revision": revision,
        "files": len(selected),
        "bytes": sum(item["bytes"] for item in selected),
        "paths": [item["path"] for item in selected],
    }
    if dry_run:
        return report
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(selected, start=1):
        destination = contained_path(output, item["path"])
        valid = (
            destination.is_file()
            and destination.stat().st_size == item["bytes"]
            and sha256(destination) == item["sha256"]
        )
        if not valid:
            cached = hf_hub_download(
                repo_id, filename=item["path"], repo_type="dataset", revision=revision, token=False
            )
            if Path(cached).stat().st_size != item["bytes"] or sha256(cached) != item["sha256"]:
                raise ValueError(f"Downloaded artifact checksum mismatch: {item['path']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            shutil.copyfile(cached, temporary)
            temporary.replace(destination)
        if extract and item.get("archive"):
            marker = destination.with_name(destination.name + ".extracted.json")
            if not marker.exists() or json.loads(marker.read_text()).get("sha256") != item["sha256"]:
                count = extract_archive(destination, output, expected_members=item.get("members"))
                write_json({"sha256": item["sha256"], "members": count}, marker)
        if progress:
            progress({"stage": "download", "completed": index, "total": len(selected), "path": item["path"]})
    write_json({"manifest": manifest, "selection": report}, output / "download_receipt.json")
    return report

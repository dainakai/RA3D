import io
import re
import tarfile

import pandas as pd
import pytest

from ra3d.datasets import select_files, extract_archive
from ra3d.io import write_table
from ra3d.review import create_app


def test_selective_download_excludes_holograms_and_other_scenes():
    manifest = {
        "files": [
            {"tier": tier, "scene": scene, "path": f"{tier}/{scene}"}
            for tier in ["features", "holograms"]
            for scene in ["seed1", "seed2"]
        ]
    }
    selected = select_files(manifest, tiers=["features"], scenes=["1"])
    assert [row["path"] for row in selected] == ["features/seed1"]
    with pytest.raises(ValueError, match="Unknown scenes"):
        select_files(manifest, tiers=["features"], scenes=["999"])


@pytest.mark.parametrize("name,link", [("../escape", False), ("/absolute", False), ("symlink", True)])
def test_archive_rejects_unsafe_members(tmp_path, name, link):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name)
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "../outside"
        else:
            info.size = 4
        tar.addfile(info, None if link else io.BytesIO(b"data"))
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "output")


def test_review_saves_raw_identity_and_rejects_untrusted_posts(tmp_path):
    pred = pd.DataFrame(
        {
            "candidate_id": ["1:2:3:10"],
            "vanished_track_id": [2],
            "survivor_track_id": [3],
            "candidate_frame": [11],
            "candidate_frame_raw": [10],
            "score": [0.9],
        }
    )
    tracks = pd.DataFrame(
        {
            "track_id": [2, 2, 3, 3],
            "frame": [9, 10, 10, 11],
            "x": [1.0, 2.0, 2.0, 3.0],
            "y": [1.0, 2.0, 2.0, 3.0],
            "z": [10.0, 10.0, 10.0, 10.0],
            "diameter_px": [2.0, 2.0, 3.0, 3.0],
        }
    )
    write_table(pred, tmp_path / "pred.parquet")
    write_table(tracks, tmp_path / "tracks.parquet")
    app = create_app(tmp_path / "pred.parquet", tmp_path / "tracks.parquet", tmp_path / "review")
    client = app.test_client()
    page = client.get("/")
    token = re.search(r'name="csrf-token" content="([^"]+)"', page.text).group(1)
    path = "/api/events/1:2:3:10/review"
    assert client.post(path, json={"label": "collision"}).status_code == 403
    assert client.post(path, json={"label": "invalid"}, headers={"X-CSRF-Token": token}).status_code == 400
    assert (
        client.post(
            path, json={"label": "collision", "note": "Visible contact"}, headers={"X-CSRF-Token": token}
        ).status_code
        == 200
    )
    exported = pd.read_csv(io.StringIO(client.get("/export.csv").text))
    assert exported.iloc[0].candidate_frame_raw == 10
    assert exported.iloc[0].candidate_frame == 11
    assert client.get("/api/events?label=unreviewed").json["filtered"] == 0
    assert client.get("/api/events/1:2:3:10").json["review"]["note"] == "Visible contact"
    assert client.get("/", headers={"Host": "attacker.example"}).status_code == 403
    recreated = create_app(
        tmp_path / "pred.parquet", tmp_path / "tracks.parquet", tmp_path / "review"
    ).test_client()
    assert recreated.get("/api/events").json["reviewed"] == 1

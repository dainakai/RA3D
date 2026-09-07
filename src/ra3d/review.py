"""Offline visual review with reconstructed event movies and durable annotations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import secrets
import sqlite3
import threading

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from flask import Flask, abort, jsonify, render_template, request, send_file, session

from .acquisition import Acquisition, Reconstructor
from .io import read_table, require_columns, require_unique, json_value
from .tracks import read_tracks
from ._core.track_images import (
    EventSpec,
    choose_center,
    point_at,
    crop_with_mean_padding,
    normalize_crop,
    finite_float,
)

LABELS = ("collision", "noncollision", "ambiguous")


def render_event_movie(event, tracks, reconstructor, destination, *, before=10, after=20):
    raw = int(event["candidate_frame_raw"])
    a = tracks.loc[tracks.track_id.eq(int(event["vanished_track_id"]))].reset_index(drop=True)
    b = tracks.loc[tracks.track_id.eq(int(event["survivor_track_id"]))].reset_index(drop=True)
    mode = "survivor_remains" if b.frame.min() <= raw - before else "merge_to_child"
    spec = EventSpec(
        "review",
        0,
        "",
        raw,
        int(event["vanished_track_id"]),
        int(event["survivor_track_id"]),
        finite_float(event.get("vanished_diameter_um")),
        finite_float(event.get("survivor_diameter_um")),
        finite_float(event.get("score")),
        mode,
    )
    images = []
    for frame in range(raw - before, raw + after + 1):
        if frame not in reconstructor.acquisition.frames.index:
            continue
        try:
            center = choose_center(spec, a, b, frame)
        except ValueError:
            continue
        intensity = reconstructor.intensity(frame, center.z)
        crop, x0, y0 = crop_with_mean_padding(intensity, center.x, center.y, 160)
        picture = Image.fromarray(normalize_crop(crop)).convert("RGB").resize((480, 480))
        draw = ImageDraw.Draw(picture)
        for track, color, name in [(a, "#f6c453", "A"), (b, "#56cfe1", "B")]:
            point = point_at(track, frame, nearest=True)
            if point is None:
                continue
            x, y = 3 * (point.x - x0), 3 * (point.y - y0)
            radius = max(5.0, 3 * point.diameter_px / 2)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
            draw.text((x + radius + 3, y), name, fill=color)
        panel = Image.new("RGB", (480, 516), "#111a25")
        panel.paste(picture, (0, 36))
        ImageDraw.Draw(panel).text(
            (12, 12), f"Frame {frame} | offset {frame - raw:+d} | z slice {center.z:.1f}", fill="white"
        )
        if frame == raw:
            ImageDraw.Draw(panel).rectangle((1, 1, 478, 514), outline="#f6c453", width=3)
        images.append(panel)
    if not images:
        raise ValueError("No synchronized image frames overlap this event")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    images[0].save(
        temporary, format="GIF", save_all=True, append_images=images[1:], duration=90, loop=0, optimize=False
    )
    temporary.replace(destination)
    return {"frames": len(images), "path": destination.name}


def create_app(predictions, tracks, workspace, *, acquisition=None, device="cpu", score_column="score"):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    frame = read_table(predictions).copy()
    track_table = read_tracks(tracks)
    require_columns(
        frame, ["vanished_track_id", "survivor_track_id", "candidate_frame", score_column], finite=True
    )
    if score_column != "score":
        frame["score"] = frame[score_column]
    if "candidate_frame_raw" not in frame:
        frame["candidate_frame_raw"] = frame.candidate_frame
    if "candidate_id" not in frame:
        frame["candidate_id"] = [
            f"0:{int(a)}:{int(b)}:{int(f)}"
            for a, b, f in zip(frame.vanished_track_id, frame.survivor_track_id, frame.candidate_frame_raw)
        ]
    frame["candidate_id"] = frame.candidate_id.astype(str)
    require_unique(frame, ["candidate_id"])
    frame = frame.sort_values(
        ["score", "candidate_frame_raw", "vanished_track_id", "survivor_track_id"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["review_rank"] = np.arange(1, len(frame) + 1)
    lookup = {str(row.candidate_id): json_value(row.to_dict()) for _, row in frame.iterrows()}
    by_track = {int(k): v for k, v in track_table.groupby("track_id", sort=False)}
    dbpath = workspace / "reviews.sqlite3"

    def connect():
        connection = sqlite3.connect(dbpath, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    with connect() as db:
        db.executescript("""PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS reviews(candidate_id TEXT PRIMARY KEY,label TEXT NOT NULL,note TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT,candidate_id TEXT NOT NULL,label TEXT NOT NULL,note TEXT NOT NULL,updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);""")
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
        ).hexdigest()
        saved = db.execute("SELECT value FROM metadata WHERE key='predictions_sha256'").fetchone()
        if saved is not None and saved["value"] != fingerprint:
            raise ValueError(
                "This review workspace belongs to different predictions; choose another workspace"
            )
        db.execute("INSERT OR IGNORE INTO metadata VALUES('predictions_sha256',?)", (fingerprint,))
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secrets.token_hex(32),
        MAX_CONTENT_LENGTH=64 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )
    render_lock = threading.Lock()
    engine = None

    def event(event_id):
        if event_id not in lookup:
            abort(404)
        return lookup[event_id]

    @app.before_request
    def local_and_csrf():
        host = request.host.split(":")[0]
        if host not in {"127.0.0.1", "localhost", "[", "::1"}:
            abort(403)
        if request.method == "POST":
            if not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), session.get("csrf", "!")):
                abort(403)
            if not request.is_json:
                abort(415)

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index():
        session["csrf"] = session.get("csrf", secrets.token_hex(24))
        return render_template("review.html", csrf=session["csrf"], has_images=bool(acquisition))

    @app.get("/api/events")
    def events():
        label = request.args.get("label", "all")
        if label not in {"all", "unreviewed", *LABELS}:
            abort(400)
        try:
            offset = max(0, int(request.args.get("offset", 0)))
            limit = min(100, max(1, int(request.args.get("limit", 40))))
        except ValueError:
            abort(400)
        with connect() as db:
            reviews = {row["candidate_id"]: dict(row) for row in db.execute("SELECT * FROM reviews")}
        selected = []
        for row in lookup.values():
            review = reviews.get(row["candidate_id"])
            status = review["label"] if review else "unreviewed"
            if label == "all" or label == status:
                selected.append({**row, "review": review})
        counts = {k: sum(r["label"] == k for r in reviews.values()) for k in LABELS}
        return jsonify(
            {
                "events": selected[offset : offset + limit],
                "filtered": len(selected),
                "total": len(frame),
                "reviewed": len(reviews),
                "counts": counts,
                "has_images": bool(acquisition),
            }
        )

    @app.get("/api/events/<event_id>")
    def detail(event_id):
        row = event(event_id)
        center = int(row["candidate_frame_raw"])
        parts = []
        for role, key in [("vanished", "vanished_track_id"), ("survivor", "survivor_track_id")]:
            values = by_track.get(int(row[key]))
            if values is None:
                continue
            values = values.loc[values.frame.between(center - 30, center + 30)]
            parts.append(
                {
                    "role": role,
                    "track_id": int(row[key]),
                    "points": json_value(values[["frame", "x", "y", "z", "diameter_px"]].to_dict("records")),
                }
            )
        with connect() as db:
            review = db.execute("SELECT * FROM reviews WHERE candidate_id=?", (event_id,)).fetchone()
        return jsonify(
            {
                "event": row,
                "tracks": parts,
                "review": dict(review) if review else None,
                "movie_url": f"/media/{hashlib.sha256(event_id.encode()).hexdigest()}.gif",
            }
        )

    @app.post("/api/events/<event_id>/review")
    def save(event_id):
        event(event_id)
        data = request.get_json()
        label = data.get("label")
        note = data.get("note", "")
        if label not in LABELS or not isinstance(note, str) or len(note) > 4000:
            abort(400)
        stamp = datetime.now(timezone.utc).isoformat()
        with connect() as db:
            db.execute(
                "INSERT INTO reviews VALUES(?,?,?,?) ON CONFLICT(candidate_id) DO UPDATE SET label=excluded.label,note=excluded.note,updated_at=excluded.updated_at",
                (event_id, label, note, stamp),
            )
            db.execute(
                "INSERT INTO history(candidate_id,label,note,updated_at) VALUES(?,?,?,?)",
                (event_id, label, note, stamp),
            )
        return jsonify({"saved": True, "candidate_id": event_id, "label": label, "updated_at": stamp})

    @app.post("/api/events/<event_id>/render")
    def render(event_id):
        nonlocal engine
        row = event(event_id)
        if not acquisition:
            return jsonify({"error": "No acquisition was supplied; trajectory review is available."}), 400
        path = workspace / "media" / (hashlib.sha256(event_id.encode()).hexdigest() + ".gif")
        try:
            with render_lock:
                if not path.exists():
                    if engine is None:
                        engine = Reconstructor(Acquisition.load(acquisition), device)
                    render_event_movie(row, track_table, engine, path)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"url": "/media/" + path.name})

    @app.get("/media/<filename>")
    def media(filename):
        stem = Path(filename).stem
        if len(stem) != 64 or any(c not in "0123456789abcdef" for c in stem) or not filename.endswith(".gif"):
            abort(404)
        path = workspace / "media" / filename
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="image/gif")

    @app.get("/export.csv")
    def export():
        with connect() as db:
            reviews = pd.read_sql_query("SELECT * FROM reviews ORDER BY candidate_id", db)
        keys = frame[
            [
                "candidate_id",
                "vanished_track_id",
                "survivor_track_id",
                "candidate_frame_raw",
                "candidate_frame",
                "score",
            ]
        ]
        output = keys.merge(reviews, on="candidate_id", how="inner", validate="one_to_one")
        data = io.BytesIO(output.to_csv(index=False).encode())
        return send_file(
            data, mimetype="text/csv", as_attachment=True, download_name="ra3d-review-labels.csv"
        )

    return app

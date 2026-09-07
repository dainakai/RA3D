# Local visual review

The review app runs entirely on your computer, inspired by the candidate queue and human labeling workflow of CollisionGifServer. It requires no Cloudflare deployment, storage bucket, remote database, CDN or external JavaScript.

## Start a review

```bash
ra3d review --run runs/my-scene --device cuda:0
```

Or provide the inputs explicitly:

```bash
ra3d review --predictions predictions.parquet --tracks tracks.parquet \
  --acquisition acquisition.yaml --workspace runs/review --port 8765
```

Open `http://127.0.0.1:8765` (or `http://localhost:8765`). Use `--port` if another local service occupies that port. The server binds only to loopback. For a remote GPU workstation, forward the port over SSH and open the forwarded localhost address. The review app is a local research tool, not a public multi-user server.

An acquisition manifest is optional for trajectory-only review. Supply it to reconstruct movies. The `demo` download includes everything needed for a small real-data example, using a subset of the released synthetic validation scene.

## Inspect and annotate

1. Select a candidate in the score-ordered queue. The header shows corrected event time and original candidate time separately.
2. Inspect the vanished track A and surviving track B. Drag the 3D view to rotate, or select XY/XZ/YZ. Use the slider or Play button to move through observed frames. Axes are scaled independently to fit the local event window; they are labeled with their numeric ranges.
3. Select **Build event movie**. The app reconstructs a 160-pixel crop from −10 through +20 raw-candidate frames, displays it at 480 pixels, and caches the GIF. The raw candidate frame has a yellow border. Camera images and reconstruction geometry come from the acquisition manifest.
4. Add a note and choose **Collision**, **Noncollision**, or **Ambiguous**. Keyboard shortcuts are 1, 2, and 3 when focus is outside the note field. Enable “Next after saving” to advance through the queue.
5. Filter by review status or export saved labels from the top-right link.

“Ambiguous” records insufficient evidence. A ranking score is not a calibrated probability, and a visually selected sample does not establish recall across an entire recording.

## Storage and reuse

The workspace contains `reviews.sqlite3` and cached `media/*.gif`. SQLite stores the current label/note and append-only revision history. Back up the workspace directory to preserve decisions. The server checks that the same workspace is not reused with different prediction contents.

The CSV export contains `candidate_id`, track pair, raw candidate frame, corrected frame, score, label, note and timestamp. Join on `candidate_id` or the raw candidate key; corrected frame alone is not stable identity. Export includes reviewed candidates only. It can support annotation and a separate active-learning design, but `prepare-training` requires exhaustive scene-level event truth rather than assuming unreviewed candidates are negative.

All page assets are packaged locally. Requests use a local session and CSRF token, and media paths are generated from candidate-ID hashes. No review labels leave your computer through the app.

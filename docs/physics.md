# Physical consistency and collision rates

The collision classifier is accompanied by physical diagnostics. These checks describe consistency of reviewed events and expected collision counts; they do not turn a selectively reviewed sample into a complete recall measurement.

## Diameter and vertical momentum

```bash
python -m ra3d.physics consistency --candidates reviewed-events.parquet \
  --tracks tracks.parquet --pixel-pitch-um 10 --fps 4000 \
  --output runs/physical-consistency.parquet
```

Filter your reviewed events as needed before calling the command. It uses the raw candidate frame, the last ten valid diameter observations at or before f−2, and the first twenty after f. The expected coalesced diameter is `(D1**3 + D2**3)**(1/3)`. Vertical velocities are medians of observed y-displacement/frame intervals over the corresponding windows. Downward y is positive. At least three observations per velocity window are required by default.

The expected post-event vertical velocity is `(D1**3*v1 + D2**3*v2)/(D1**3 + D2**3)`. Missing evidence produces NaN and explicit availability/count columns. Output JSON reports sample counts, errors and correlations. No new classification labels are assigned by this diagnostic.

## Hall-corrected gravitational rate

```bash
python -m ra3d.physics rate --particles measured-particles.parquet \
  --diameter-column final_diameter_um --recorded-frames 10917 \
  --volume-cm3 10.71644672 --fps 4000 \
  --output runs/gravitational-pair-counts.parquet
```

Use the **frame-wise measured particle table**, not one row per trajectory. Replace the exposure and volume with your actual detector support. The default vanishing-endpoint policy excludes the last 13 recorded frames, giving 10,904 effective frames for the example above. The estimator uses 5 µm density bins over 25≤D<300 µm, reports unordered 25 µm diameter-pair bins, and applies a factor of one half within a pair-bin diagonal to avoid double counting.

The implementation preserves the paper's Davis–Jonas–Hall table interpolation and Beard terminal-velocity calculation at 900 hPa and 307.182512 K. Its numerical table comes from the BSD-licensed [SCALE-SDM source archive](https://doi.org/10.5281/zenodo.3841674), accompanying [Shima et al. (2020)](https://doi.org/10.5194/gmd-13-4107-2020). Coalescence efficiency is one. The table, atmospheric metadata, and original SCALE notice are bundled under `config/physics`; source-symbol hashes are recorded separately.

Eleven temporal density blocks are resampled 4,000 times with seed 260724. Predictive variance is the expected Poisson count plus the bootstrap variance of the conditional mean. The API exposes block count, exposure policy, seed and kernel for custom analyses. Pooled results require pooling bootstrap counts before calculating uncertainty; summing per-cell standard deviations is not equivalent.

Visually confirmed event counts are lower bounds when missed detections have not been measured. Comparing these counts to a gravitational kernel is a consistency check, not an efficiency-corrected experimental collision-kernel measurement.

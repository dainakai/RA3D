# Synthetic scenes and hologram generation

RA3D includes the numerical scene generator and a Python/PyTorch angular-spectrum renderer. Generating labels is separate from rendering images, so you can study particle dynamics without producing thousands of holograms.

## Regenerate a paper scene

From the repository root:

```bash
python -m ra3d.simulation.generate \
  --project-root . --out-dir runs/synthetic-seed260740 \
  --theme collision_benchmark_260715_halfmix25_v1 \
  --frames 6000 --seed 260740 --target-particles 200 \
  --target-planned-collisions 2000 --half-mixture-25um \
  --half-mixture-coverage-collisions 1000 \
  --half-mixture-candidates-per-pair 6000 \
  --half-mixture-parent-lead-frames 1500 \
  --diameter-density-csv src/ra3d/config/simulation/diameter_distribution_260715.csv \
  --velocity-quantiles-csv src/ra3d/config/simulation/velocity_quantiles_260715.csv \
  --disable-natural-collisions --small-particle-y-jitter --refresh-source
```

Change only `--seed` to generate the other five paper scenes. The full seed260740 port was checked against the original: all 2,000 events, 200 initial particles, 2,102,484 per-frame particle records and 8,378 trajectory records matched at rtol 1e-10 / atol 1e-8, including matching categorical columns.

The generator samples diameters from the bundled experimental distribution, horizontal/axial velocities from the bundled inverse CDF, and vertical velocities from Beard terminal velocity minus 0.26 m/s upward flow. Small droplets receive the specified bounded per-particle vertical jitter. Particle replenishment and collision parents begin outside the padded propagation boundary. Coalescence conserves volume and momentum. Half the planned collisions follow the empirical chase distribution and half cover stratified 25 µm diameter pairs, speed tiers and impact-parameter tiers.

`labels/` contains complete particle/event truth. `metadata/` records parameters, sampled distributions, pair coverage and source snapshots. Use a fresh output directory for a changed recipe. Arbitrarily shortened scenes cannot necessarily accommodate 1,500-frame collision-parent lead times and 2,000 events; adjust planning parameters together for small custom simulations.

## Render synchronized images

```bash
python -m ra3d.simulation.render \
  --particles runs/synthetic-seed260740/labels/particles.csv \
  --output runs/synthetic-seed260740/holograms --device cuda:0 \
  --start-frame 0 --end-frame 6000 --resume
```

`--end-frame` is exclusive. Each frame produces `primary/000001.png` and `secondary/000001.png` for frame 0, and so on. `--resume` skips already present pairs. The renderer records optical settings in `render.json`.

The reference optical simulation uses a 1,536-pixel propagation plane with a 1,024-pixel central crop, 10 µm pixel pitch, 0.6328 µm wavelength and gray background 77.5. Opaque particle disks are ordered by depth; the wavefront propagates between particles before being observed at the two camera planes. This preserves multiple-particle transmission rather than summing independent particle images. `RenderOptions` provides explicit Python overrides for new optical experiments.

The Python renderer differs from the original Julia output by at most one gray level in the four verified reference images, due to floating-point/quantization details. Use the released original PNG shards for exact image replay. The generator is noise-free; performance on a new experimental imaging system requires separate validation.

The scene geometry Z origin used by the simulator is separate from the reconstruction-slice origin. Use the released acquisition YAML for the paper's reconstructed measurements; do not copy camera-front distances directly into collision-table `z_um`.

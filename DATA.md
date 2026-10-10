# Data accompanying this repository

Two kinds of data are involved. Small, human-readable results are tracked in
this repository; raw per-case simulation logs and large model parameters are
distributed as a separate archive because of their size.

## Tracked in this repository

| Path | Content |
|---|---|
| `Paper/tables/*.csv`, `*.json` | processed statistics used by the manuscript tables |
| `Logs/manuscript_revision/` | per-case rows for the matched reruns (damage-mask hash, seed, event indicator, timing). Contains the offset-zero ablation cohort (`ablation.csv`, 50 configurations per variant) and the retained repaired cohort (`ablation_repair.csv`, 50 further configurations per variant); the N=1000 ablation statistics combine them. `repair_ablation_provenance.py` validates both. |
| `Paper/figs/**/*.stats.json` | figure statistics actually plotted |
| `Paper/figs/fig2_cache/*.npz`, `fig3_cache/*.npz` | trajectory caches used to draw the figures |
| `Configurations/swarm_positions_*.xlsx` | initial swarm configurations |
| `Pretrained_model/model.pt` | ML-DAGL weights required by that baseline |

## Data archive

Archived on Zenodo under `<ZENODO DOI>` (DOI to be minted before submission;
the archive file is `morphica-data-archive.tar.gz`). Code repository:
<https://github.com/lytxzt/morphica-swarm-repair>.
The archive contains the per-case evidence behind every published number:
**498 MB raw, 215 MB compressed** as built by `tools/pack_data_archive.sh`
(measured; 1517 entries). Dropping `Meta_Learning_Results/` — needed only to
re-run the CR-MGC baseline — leaves roughly 110 MB compressed.

| Path | Content | Size |
|---|---|---|
| `Logs/algorithm_sweep/` | per-case result logs and K(t) trajectories for every benchmark cell (all methods, all scales, CR sensitivity, trajectory subsets) | ~352 MB |
| `Logs/ugv/manuscript_matrix/` | frozen 200-scenario ground-swarm matrix: scenario ids, destruction-mask hashes, per-run metrics | ~2.7 MB |
| `Logs/ugv/cr80_sensitivity/` | ridge-only CR=80 sensitivity runs | ~1.2 MB |
| `Logs/ring_center_test/` | 45-node ring–centre diagnostic trajectory (Fig. 3d) | ~1.1 MB |
| `Logs/ugv/render_demo/case_mixed_random_MORPHICA_N100_D50_r50_cr120_w1.2_seed7.json`, `.../terrain/mixed_N100_s10013` | representative terrain replay behind Fig. 4a | ~1.8 MB |
| `Meta_Learning_Results/` | CR-MGC meta-parameters (`meta_parameters/*.npy`), needed to re-run that baseline | ~109 MB |
| `Logs/ugv/cases/`, `Logs/ugv/terrain_trajectories/` | optional: UGV demo cases used only by auxiliary UGV figures | ~31 MB |

Deliberately **not** archived (legacy, intermediate or presentation material that
backs no published number): `Logs/archive/` (358 MB), `Logs/old_runs/` (163 MB),
`Logs/running_logs/` (53 MB), `Logs/ugv/cr_scenarios_full.json` (112 MB, the
pre-selection scenario scan), the remaining `Logs/ugv/render_demo/` video and
render frames (47 MB), and `Logs/e6_positions/`.

### Unpack

Unpack at the repository root so that the paths above sit next to `README.md`:

```bash
tar -xzf morphica-data-archive.tar.gz -C /path/to/this/repository
sha256sum -c morphica-data-archive.tar.gz.sha256
```

### Rebuild the archive

From a full working copy (the one that still contains `Logs/` and
`Meta_Learning_Results/`):

```bash
bash tools/pack_data_archive.sh morphica-data-archive.tar.gz
```

Damage masks are deterministic in the case index: for a case with seed index
`c` at scale `N` with `D` removals, the removal set is
`random.Random(c + 1725).sample(range(N), D)`, which matches the historical
runner's `config_random_seed[c] + 1718` with `config_random_seed = range(7, 10007)`.
`Logs/manuscript_revision/*.manifest.json` records the same rule and the
SHA-256 of every mask used in the reruns.

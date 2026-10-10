# Data accompanying this repository

Two kinds of data are involved. Small, human-readable results are tracked in this
repository; raw per-case logs and large model parameters are packaged in a
separate local archive because of their size.

## Tracked in this repository

| Path | Content |
|---|---|
| `data/control_suites/*.csv` | per-case rows for the matched control suites (suite, case, seed, damage-mask SHA-256, event indicator, timing) |
| `data/control_suites/*.manifest.json` | suite metadata: case count, cutoffs, seed rules, source hashes |
| `data/control_suites/diagnostics/` | connected-ring diagnostic replay (trajectory, mask hash, artifact hash) |
| `data/configs/swarm_positions_*.xlsx` | initial swarm configurations |
| `data/pretrained/model.pt` | ML-DAGL weights required by that method |

## Data archive

The archive is prepared locally and **has not been publicly released**. A
download link will be added when it is published. Its three files are
`morphica-data-archive.tar.gz`, `morphica-data-archive.tar.gz.sha256` and
`morphica-data-archive.tar.gz.contents.txt`.

The code and tracked inputs support fresh main benchmarks. Verification and
reruns of matched control suites use case lists from the raw logs and require
the local archive; CR-MGC reruns also require its meta-parameters.

Together with the tracked control-suite results, the archive contains the
per-case evidence for the manuscript:
**498 MB raw, 215 MB compressed** as built by `tools/pack_data_archive.sh`.

| Path in the archive | Content | Size |
|---|---|---|
| `data/logs/algorithm_sweep/` | per-case logs and K(t) trajectories for every benchmark cell (all methods, all scales, communication-range sensitivity, trajectory subsets) | ~352 MB |
| `data/logs/ugv/benchmark_matrix/` | frozen 200-scenario ground-swarm matrix: scenario ids, destruction-mask hashes, per-run metrics | ~2.7 MB |
| `data/logs/ugv/cr80_sensitivity/` | ridge-only CR=80 sensitivity runs | ~1.2 MB |
| `data/logs/ring_center_test/` | connected-ring diagnostic trajectory | ~1.1 MB |
| `data/logs/ugv/render_demo/...` | representative terrain replay and terrain directory | ~1.8 MB |
| `data/meta_parameters/` | CR-MGC meta-parameters, needed to re-run that method | ~109 MB |
| `data/logs/ugv/cases/`, `data/logs/ugv/terrain_trajectories/` | optional: ground-swarm demo cases used by auxiliary figures | ~31 MB |

Deliberately **not** archived (superseded, intermediate or presentation material
that backs no published number): the superseded sweep trees, the pre-selection
scenario scans, and the rendered ground-swarm videos and frames.

### Unpack

Unpack at the repository root so that the paths above sit next to `README.md`:

```bash
cd /path/to/this/repository
sha256sum -c morphica-data-archive.tar.gz.sha256
tar -xzf morphica-data-archive.tar.gz
```

Place the local archive and its checksum in the repository root before running
these commands. The checksum records only the archive's filename, so it also
works after moving the files to another directory.

### Rebuild the archive

From a full working copy (one that still contains the raw log tree and the
meta-parameters):

```bash
bash /path/to/this/repository/tools/pack_data_archive.sh morphica-data-archive.tar.gz /path/to/full-working-copy
```

### Damage masks

Damage is deterministic in the case index, so every method, scale and
communication radius sees identical damage for a given case:

```python
from tools.damage_mask import damage_mask
damage_mask(n=1000, d=500, case=0)     # removal set for case 0
```

The rule is `random.Random(case + 1725).sample(range(N), D)`, matching the seed
schedule of the original sweep runner. Each tracked per-case row stores the
SHA-256 of the mask it used, and the manifests record the same rule.

## Ablation cohorts

The N=1000 operator ablations combine two cohorts of 50 configurations per
variant, both on D=500: `ablation.csv` (the offset-zero cohort) and
`ablation_cohort_b.csv` (cases 100–149, checked against the corresponding
historical no-P/no-S logs). Use
`python experiments/run_ablation_cohort_b.py --verify-archived` to verify that
cohort. Archived source hashes describe the
recorded runs; new runs store their own manifests under the selected output
directory.

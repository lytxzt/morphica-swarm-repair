# MORPHICA — Morphogenetic Intelligence for Collective Assembly

Source-free morphogenetic repair for fragmented robot swarms. Severe damage
disconnects a swarm and destroys the communication substrate that collective
computation depends on; MORPHICA rebuilds that substrate without new agents,
without global information and without training.

The current contact graph defines morphogenetic domains. Component-confined
positional, front and boundary summaries drive concurrent **R**etraction,
**P**olarization and **S**preading; the movement they produce creates the
communication edges that regenerate the graph on which the next signal state is
computed.

This repository is the code and processed-data release accompanying the
manuscript: the controller, the benchmark and control-suite runners, the
operator-ablation implementations, the comparison baselines, the table builders,
the figure scripts, the processed statistics used by the manuscript, and the
inputs needed to reproduce them. Raw per-case simulation logs and large model
parameters are distributed separately — see [DATA.md](DATA.md).

## Layout

```
Configurations.py Environment.py Swarm.py Utils.py   simulation core + controller interface
DIRA_Algorithm/                                      MORPHICA and relay-limited MORPHICA
Previous_Algorithm/ DAGL_Algorithm/                  comparison baselines
Experiment_Statistic.py                              main sweep runner (all methods, one interface)
tools/run_manuscript_evidence.py                     matched reruns: null, ablations, observation stress, relay
tools/test_manuscript_evidence.py                    regression checks
tools/build_manuscript_supplement.py                 builds the supplementary tables
Paper/plot_fig*.py, Paper/plot_supp_*.py             manuscript figures
Paper/tables/                                        processed statistics used by the manuscript
Paper/figs/                                          figure statistics + trajectory caches
Logs/manuscript_revision/                            per-case rows for the matched reruns
UGV/                                                 terrain-constrained ground-swarm benchmark
Configurations/                                      initial swarm configurations (xlsx)
Pretrained_model/                                    ML-DAGL weights (required by that baseline)
```

## Install

CPU-only pipeline — controller, benchmarks, table builders, figures:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Comparison baselines (GPU, optional):

```bash
pip install -r requirements-baselines.txt
```

The controller is CPU-only and training-free. The published baseline runs used
torch 2.6.0+cu124 on NVIDIA RTX 4080 SUPER GPUs.

## Reproduce

Raw per-case logs are not stored in git. Unpack the data archive first (see
[DATA.md](DATA.md)) so that `Logs/algorithm_sweep/`, `Logs/ugv/` and
`Meta_Learning_Results/` sit next to this README.

```bash
# matched control suites (null, operator ablations, observation stress, finite relay)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_manuscript_evidence.py --suite null --workers 8

python tools/test_manuscript_evidence.py        # regression checks
python tools/build_manuscript_supplement.py     # regenerate Paper/tables/*.csv
python Paper/plot_fig2.py --panel b             # regenerate one figure panel
```

`tools/run_manuscript_evidence.py` resumes completed rows and rejects changes to
code or geometry hashes, so a partially completed suite can be continued safely.
Every per-case row stores its damage-mask SHA-256, seed and event indicator.

## Data

Small results are tracked here; raw logs and large parameters are archived —
see [DATA.md](DATA.md) for contents, sizes, unpacking and checksums.

Repository: <https://github.com/lytxzt/morphica-swarm-repair>

## Provenance and attribution

All code here was written by the authors. The comparison methods are the
authors' own reimplementations of published algorithms, driven through one
shared interface; the original publications are listed in
[PROVENANCE.md](PROVENANCE.md) and cited in the manuscript. No third-party
source code or publisher material is redistributed.

## Licence and citation

Released under the MIT licence — see [LICENSE](LICENSE). If you use this code,
please cite the manuscript; the full citation and DOI will be added here on
publication. Code: <https://github.com/lytxzt/morphica-swarm-repair>.

# MORPHICA — Morphogenetic Intelligence for Collective Assembly

Source-free morphogenetic repair for fragmented robot swarms. Severe damage
disconnects a swarm and destroys the communication substrate that collective
computation depends on; MORPHICA rebuilds it without new agents, without global
information and without training.

The current contact graph defines morphogenetic domains. Component-confined
positional, front and boundary summaries drive concurrent **R**etraction,
**P**olarization and **S**preading; the movement they produce creates the
communication edges that regenerate the graph on which the next signal state is
computed.

This repository contains the controller, the simulation core, the benchmark and
control-suite runners, the comparison methods, the diagnostic scripts, the
per-case results of the control suites, and the inputs needed to reproduce them.
Raw per-case logs and the larger model parameters are distributed separately —
see [DATA.md](DATA.md).

Repository: <https://github.com/lytxzt/morphica-swarm-repair>

## Layout

```
Configurations.py Environment.py Swarm.py Utils.py   simulation core and controller interface
controllers/                                         MORPHICA and the relay-limited variant
baselines/                                           comparison methods (see PROVENANCE.md)
baselines/dagl/                                      ML-DAGL reimplementation
experiments/run_main_sweep.py                        main sweep runner (all methods, one interface)
experiments/run_control_suites.py                    matched control suites: null, ablations, observation stress, relay
experiments/run_ablation_cohort_b.py                 second ablation cohort on a fixed case set
experiments/ring_center_diagnostic.py                connected-ring diagnostic
experiments/ring_center_replay.py                    explicit-mask replay of the ring-centre diagnostic
experiments/dump_final_positions.py                  terminal positions for the topology analysis
experiments/degree_distribution.py                   k-hop degree statistics
experiments/run_single_case.py                       single-case replay
tests/test_control_suites.py                         regression checks
tools/damage_mask.py                                 canonical damage-mask rule
tools/make_swarm_positions.py                        generate initial-configuration spreadsheets
tools/pack_data_archive.sh                           build the data archive
data/configs/                                        initial swarm configurations (xlsx)
data/control_suites/                                 per-case rows, manifests and diagnostics
data/pretrained/                                     ML-DAGL weights
data/logs/                                           unpack the data archive here
ugv/                                                 terrain-constrained ground-swarm benchmark
```

## Install

CPU-only pipeline — controller, benchmarks, control suites and diagnostics:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Comparison methods (GPU, optional):

```bash
pip install -r requirements-baselines.txt
```

The controller is CPU-only and training-free. The published baseline runs used
torch 2.6.0+cu124 on NVIDIA RTX 4080 SUPER GPUs.

## Reproduce

Raw per-case logs are not stored in git. Unpack the data archive first (see
[DATA.md](DATA.md)) so that `data/logs/` and `data/meta_parameters/` are
populated.

```bash
# matched control suites: null (random waypoints), ablations, observation stress, finite relay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_control_suites.py --suite null --workers 8

python tests/test_control_suites.py            # regression checks
python experiments/run_main_sweep.py --algorithm MORPHICA --num-agents 1000 \
    --destroyed-values 500 --max-cases 50 --output-dir data/logs/morphica_check
```

`experiments/run_control_suites.py` resumes completed rows and rejects changes
to code or geometry hashes, so a partially completed suite can be continued
safely. Every per-case row stores its damage-mask SHA-256, seed and event
indicator.

## Data

Small results are tracked here; raw logs and large parameters are archived — see
[DATA.md](DATA.md) for contents, sizes, unpacking and checksums.

## Provenance and attribution

All code here was written by the authors. The comparison methods are the
authors' own reimplementations of published algorithms, driven through one
shared interface; the original publications are listed in
[PROVENANCE.md](PROVENANCE.md). No third-party source code or publisher material
is redistributed.

## Licence and citation

Released under the MIT licence — see [LICENSE](LICENSE). If you use this code,
please cite the manuscript; the full citation and DOI will be added here on
publication.

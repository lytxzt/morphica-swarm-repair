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
Raw per-case logs and the larger model parameters belong to a separate archive,
whose contents and availability are described in [DATA.md](DATA.md).

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

Use Python 3.10 or 3.11 for both environments below. The CPU pipeline also
supports Python 3.8 and 3.9. MORPHICA, HERO, CEN and SIDR run with the core
dependencies alone:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Learned and graph-optimisation comparison methods (Python 3.9–3.11):

```bash
pip install -r requirements-baselines.txt
```

The controller is CPU-only and training-free. The published baseline runs used
torch 2.6.0+cu124 on NVIDIA RTX 4080 SUPER GPUs.

## Reproduce

Raw per-case logs are not stored in git, and the separate data archive has not
yet been published. With a local copy of that archive, unpack it first (see
[DATA.md](DATA.md)) so that `data/logs/` and `data/meta_parameters/` are populated.
The regression checks and fresh main sweep below run without the archive;
verification of the recorded suites and their matched reruns require it.

```bash
# Verify the archived control results (no simulation or file changes)
for suite in null ablation nonideal relay; do
    python experiments/run_control_suites.py --suite "$suite" --verify-archived
done
python experiments/run_ablation_cohort_b.py --verify-archived

python tests/test_control_suites.py            # regression checks
python tests/test_release.py                  # installation and archive checks

# Run a fresh main benchmark
python experiments/run_main_sweep.py --algorithm MORPHICA --num-agents 1000 \
    --destroyed-values 500 --max-cases 50 --output-dir data/logs/morphica_check

# Rerun a control suite into a separate output directory
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_control_suites.py \
    --suite null --workers 8 --output-dir outputs/control_suites
python experiments/run_ablation_cohort_b.py --workers 8 --output-dir outputs/control_suites
```

Both control runners default to `outputs/control_suites/`. They resume new runs
only when the code and geometry hashes match their run manifests. The archived
CSVs and manifests in `data/control_suites/` retain the recorded results and
their original source information; `--verify-archived` checks task coverage,
damage masks, geometry and connectivity events. Every per-case row stores its
damage-mask SHA-256, seed and event indicator.

## Data

Small results are tracked here; raw logs and large parameters are archived locally — see
[DATA.md](DATA.md) for contents, sizes, unpacking and checksums.

## Provenance and attribution

All code here was written by the authors. The comparison methods are the
authors' own reimplementations of published algorithms, driven through one
shared interface; the original publications are listed in
[PROVENANCE.md](PROVENANCE.md). No third-party source code or publisher material
is redistributed.

## Licence and citation

Released under the MIT licence — see [LICENSE](LICENSE). Cite the software
using [CITATION.cff](CITATION.cff). The
[v1.0.1 tag](https://github.com/lytxzt/morphica-swarm-repair/tree/v1.0.1)
identifies the code snapshot. Data availability is recorded in [DATA.md](DATA.md).

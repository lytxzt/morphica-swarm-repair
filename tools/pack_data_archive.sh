#!/usr/bin/env bash
# Build the data archive that accompanies this repository.
#
# The archive carries the per-case evidence behind every published number, not
# the whole working log tree: superseded sweeps, pre-selection scans and
# rendered media are excluded (see DATA.md). Run it from a full working copy,
# i.e. one that still contains Logs/ and Meta_Learning_Results/. Pass that
# working copy as the second argument when using this script from the release.
set -euo pipefail

OUT=${1:-morphica-data-archive.tar.gz}
OUT=$(realpath -m "$OUT")
ROOT=${2:-$(cd "$(dirname "$0")/.." && pwd)}
ROOT=$(cd "$ROOT" && pwd)
cd "$ROOT"

EVIDENCE=(
  "Logs/algorithm_sweep"        # per-case logs and K(t) trajectories for every benchmark cell
  "Logs/ugv/manuscript_matrix"  # frozen 200-scenario ground-swarm matrix, masks, metrics
  "Logs/ugv/cr80_sensitivity"   # ridge-only CR=80/120 sensitivity
  "Logs/ring_center_test"       # connected-ring diagnostic trajectory
  "Logs/ugv/render_demo/case_mixed_random_MORPHICA_N100_D50_r50_cr120_w1.2_seed7.json"
  "Logs/ugv/render_demo/terrain/mixed_N100_s10013"
  "Meta_Learning_Results/meta_parameters"  # CR-MGC meta-parameters, needed to re-run that method
)
OPTIONAL=(
  "Logs/ugv/cases"
  "Logs/ugv/terrain_trajectories"
)

for p in "${EVIDENCE[@]}"; do
  [ -e "$p" ] || { echo "missing required path: $p" >&2; exit 1; }
done

PATHS=("${EVIDENCE[@]}")
for p in "${OPTIONAL[@]}"; do [ -e "$p" ] && PATHS+=("$p"); done

echo "packing ${#PATHS[@]} paths (~$(du -sch "${PATHS[@]}" 2>/dev/null | tail -1 | cut -f1) raw)"
tar --transform='s|^Logs/|data/logs/|;s|^Meta_Learning_Results/meta_parameters|data/meta_parameters|;s|^data/logs/ugv/manuscript_matrix|data/logs/ugv/benchmark_matrix|' \
    -czf "$OUT" "${PATHS[@]}"
tar -tzf "$OUT" > "$OUT.contents.txt"
(cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")") > "$OUT.sha256"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), plus .contents.txt and .sha256"

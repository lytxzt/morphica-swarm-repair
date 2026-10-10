#!/usr/bin/env bash
# Build the data archive that accompanies this repository.
#
# The archive carries the per-case evidence behind every published number, not
# the whole working Logs/ tree: legacy sweeps, pre-selection scans and rendered
# media are excluded (see DATA.md). Run from a full working copy.
set -euo pipefail

OUT=${1:-morphica-data-archive.tar.gz}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

EVIDENCE=(
  "Logs/algorithm_sweep"        # per-case logs + K(t) trajectories for every benchmark cell
  "Logs/ugv/manuscript_matrix"  # frozen 200-scenario matrix, destruction masks, metrics
  "Logs/ugv/cr80_sensitivity"   # ridge-only CR=80/120 sensitivity
  "Logs/ring_center_test"       # 45-node ring-centre diagnostic (Fig. 3d)
  "Logs/ugv/render_demo/case_mixed_random_MORPHICA_N100_D50_r50_cr120_w1.2_seed7.json"
  "Logs/ugv/render_demo/terrain/mixed_N100_s10013"
  "Meta_Learning_Results"       # CR-MGC meta-parameters (needed to re-run that baseline)
)
OPTIONAL=(
  "Logs/ugv/cases"              # UGV demo cases, used only by auxiliary UGV figures
  "Logs/ugv/terrain_trajectories"
)

for p in "${EVIDENCE[@]}"; do
  [ -e "$p" ] || { echo "missing required path: $p" >&2; exit 1; }
done

PATHS=("${EVIDENCE[@]}")
for p in "${OPTIONAL[@]}"; do [ -e "$p" ] && PATHS+=("$p"); done

echo "packing ${#PATHS[@]} paths (~$(du -sch "${PATHS[@]}" 2>/dev/null | tail -1 | cut -f1) raw)"
tar -czf "$OUT" "${PATHS[@]}"
tar -tzf "$OUT" > "$OUT.contents.txt"
sha256sum "$OUT" > "$OUT.sha256"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)), plus .contents.txt and .sha256"

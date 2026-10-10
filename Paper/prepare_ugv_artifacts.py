#!/usr/bin/env python3
"""Create a non-destructive, provenance-checked UGV paper artifact bundle.

The benchmark logs remain the source of truth under ``Logs/ugv``.  This tool
copies only the frozen manuscript matrix, its setup-only selection inputs, the
matched CR sensitivity result, and the selected visual replay inputs into an
ignored paper-local directory.  It writes SHA-256 values for both source and
copied files so figures and supplementary material can be traced back to the
frozen benchmark outputs.

Usage:
    python Paper/prepare_ugv_artifacts.py
    python Paper/prepare_ugv_artifacts.py --output /tmp/ugv_paper_bundle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "Paper" / "artifacts" / "ugv"

# Keep this list deliberately small.  Pilot/stage runs are not manuscript
# evidence and must not be accidentally used when producing paper figures.
ARTIFACTS = (
    ("final_matrix/ugv_summary.csv",
     "Logs/ugv/manuscript_matrix/metrics/ugv_summary.csv",
     "Frozen 1,200-case UGV manuscript matrix."),
    ("final_matrix/benchmark_manifest.json",
     "Logs/ugv/manuscript_matrix/metrics/benchmark_manifest.json",
     "Exact code/configuration provenance for the frozen matrix."),
    ("final_matrix/resume_fingerprint.json",
     "Logs/ugv/manuscript_matrix/metrics/ugv_summary_resume_fingerprint.json",
     "Resume compatibility fingerprint for the frozen matrix."),
    ("selection/setup_scan.csv",
     "Logs/ugv/setup_scan_manuscript/metrics/setup_scan.csv",
     "Algorithm-free scan used to select initially fragmented scenarios."),
    ("selection/unbiased_selection_v2.json",
     "Logs/ugv/unbiased_selection_v2.json",
     "Deterministic cross-D seed selection."),
    ("selection/scenario_manifest_v2.json",
     "Logs/ugv/scenario_manifest_v2.json",
     "Scenario list used by every algorithm in the final matrix."),
    ("cr_sensitivity/ugv_summary.csv",
     "Logs/ugv/cr80_sensitivity/metrics/ugv_summary.csv",
     "Matched ridge CR=80/120 sensitivity output."),
    ("cr_sensitivity/benchmark_manifest.json",
     "Logs/ugv/cr80_sensitivity/metrics/benchmark_manifest.json",
     "Configuration provenance for CR sensitivity."),
    ("visual_replay/mixed_seed7/MORPHICA.json",
     "Logs/ugv/render_demo/case_mixed_random_MORPHICA_N100_D50_r50_cr120_w1.2_seed7.json",
     "Kinematic replay input: MORPHICA."),
    ("visual_replay/mixed_seed7/CEN.json",
     "Logs/ugv/render_demo/case_mixed_random_CEN_N100_D50_r50_cr120_w1.2_seed7.json",
     "Kinematic replay input: CEN."),
    ("visual_replay/mixed_seed7/HERO.json",
     "Logs/ugv/render_demo/case_mixed_random_HERO_N100_D50_r50_cr120_w1.2_seed7.json",
     "Kinematic replay input: HERO."),
    ("visual_replay/mixed_seed7/terrain/heightmap.npy",
     "Logs/ugv/render_demo/terrain/mixed_N100_s10013/heightmap.npy",
     "Terrain height field used by all three replay trajectories."),
    ("visual_replay/mixed_seed7/terrain/costmap.npy",
     "Logs/ugv/render_demo/terrain/mixed_N100_s10013/costmap.npy",
     "Terrain cost field used by all three replay trajectories."),
    ("visual_replay/mixed_seed7/terrain/obstacle_mask.npy",
     "Logs/ugv/render_demo/terrain/mixed_N100_s10013/obstacle_mask.npy",
     "Hard-hazard mask used by all three replay trajectories."),
    ("visual_replay/mixed_seed7/terrain/metadata.json",
     "Logs/ugv/render_demo/terrain/mixed_N100_s10013/metadata.json",
     "Terrain bounds and generation metadata."),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_checked(source: Path, destination: Path) -> dict[str, object]:
    source_hash = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination_hash = sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(
                f"Refusing to overwrite different artifact: {destination}. "
                "Choose a new --output directory or remove that bundle explicitly."
            )
        action = "verified_existing"
    else:
        shutil.copy2(source, destination)
        destination_hash = sha256(destination)
        if destination_hash != source_hash:
            raise RuntimeError(f"Checksum mismatch after copying {source}")
        action = "copied"
    return {
        "source": str(source.relative_to(ROOT)),
        "bundle_path": str(destination),
        "sha256": source_hash,
        "bytes": source.stat().st_size,
        "action": action,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Bundle directory (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()
    output = args.output.resolve()

    manifest_entries = []
    for bundle_rel, source_rel, purpose in ARTIFACTS:
        source = ROOT / source_rel
        if not source.is_file():
            raise FileNotFoundError(f"Required artifact is missing: {source}")
        entry = copy_checked(source, output / bundle_rel)
        entry["purpose"] = purpose
        manifest_entries.append(entry)

    manifest = {
        "bundle_schema": 1,
        "source_of_truth": "Original files remain under Logs/ugv; this is a read-only paper bundle.",
        "excluded": [
            "pilot_*", "stageA_*", "stageB_*", "stageC_*",
            "final_matrix_CR120*", "main_matrix_CR120", "unbiased_final_*",
        ],
        "artifacts": manifest_entries,
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"UGV paper artifact bundle: {output}")
    print(f"Artifacts: {len(manifest_entries)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

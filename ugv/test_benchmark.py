"""Smoke tests for ugv/run_ugv_benchmark.py.

Quick validation that catches serialisation bugs (e.g. frozenset in JSON),
seed parsing, and basic end-to-end flow without running a full sweep.

Run:  python ugv/test_benchmark.py
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

BENCHMARK = 'ugv/run_ugv_benchmark.py'


def _run_bench(*args):
    """Run the benchmark script and return (returncode, stdout, stderr)."""
    rc = subprocess.run(
        [sys.executable, BENCHMARK] + list(args),
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    )
    return rc.returncode, rc.stdout, rc.stderr


# ═══════════════════════════════════════════════════════════════════════════
# CLI validation
# ═══════════════════════════════════════════════════════════════════════════
def test_seed_range_out_of_bounds_rejected():
    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '10',
        '--terrains', 'flat',
        '--seed-range', '0', '99999',
        '--output-dir', '/tmp/_ugv_smoke_',
    )
    assert rc != 0, f'expected non-zero, got {rc}'
    assert 'out of bounds' in stderr, f'got: {stderr}'


def test_seed_range_reversed_rejected():
    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '10',
        '--terrains', 'flat',
        '--seed-range', '4', '0',
        '--output-dir', '/tmp/_ugv_smoke_',
    )
    assert rc != 0, f'expected non-zero, got {rc}'
    assert 'must be <=' in stderr, f'got: {stderr}'


def test_negative_seed_in_list_rejected():
    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '10',
        '--terrains', 'flat',
        '--seeds', '-1',
        '--output-dir', '/tmp/_ugv_smoke_',
    )
    assert rc != 0, f'expected non-zero, got {rc}'


def test_seeds_and_range_mutually_exclusive():
    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '10',
        '--terrains', 'flat',
        '--seeds', '0', '1',
        '--seed-range', '0', '4',
        '--output-dir', '/tmp/_ugv_smoke_',
    )
    assert rc != 0, f'expected non-zero, got {rc}'


def test_missing_algorithms_rejected():
    rc, _, _ = _run_bench(
        '--terrains', 'flat',
        '--output-dir', '/tmp/_ugv_smoke_',
    )
    assert rc != 0, f'expected non-zero, got {rc}'


# ═══════════════════════════════════════════════════════════════════════════
# manifest + end-to-end smoke
# ═══════════════════════════════════════════════════════════════════════════
def test_manifest_is_valid_json_and_serialisable(tmp_path):
    """A single-case benchmark must produce a valid manifest.json."""
    out = tmp_path / 'smoke'
    rc, stdout, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '10',
        '--terrains', 'flat',
        '--seed-range', '0', '0',
        '--output-dir', str(out),
        '--omega-max', '1.2',
    )
    # May succeed or fail depending on algorithm availability—either is fine
    # as long as the manifest was written and is valid JSON.
    manifest_path = out / 'metrics' / 'benchmark_manifest.json'
    assert manifest_path.exists(), (
        f'manifest not created\nstdout:\n{stdout}\nstderr:\n{stderr}')
    with open(manifest_path) as f:
        mf = json.load(f)
    assert 'git_commit' in mf
    assert 'algorithms' in mf
    assert 'safety_controller' in mf
    assert 'algo_modes' in mf
    assert isinstance(mf['one_shot_modes'], list), (
        f'one_shot_modes should be a list, got {type(mf["one_shot_modes"])}')
    assert 'config' in mf
    assert 'communication_range' in mf['config']
    assert 'git_worktree_dirty' in mf
    assert 'git_status' in mf
    if mf['git_worktree_dirty']:
        # git_status must be a non-empty list when dirty
        assert isinstance(mf['git_status'], list)
        assert len(mf['git_status']) > 0
    assert 'constant_speed' in mf['config']
    assert 'git_worktree_dirty' in mf


def test_communication_range_flag_accepted(tmp_path):
    """--communication-range (int) must be accepted and recorded in manifest."""
    out = tmp_path / 'cr_test'
    rc, stdout, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '10',
        '--terrains', 'flat',
        '--seed-range', '0', '0',
        '--communication-range', '80',
        '--output-dir', str(out),
    )
    assert 'unrecognized arguments' not in stderr, f'got: {stderr}'
    manifest_path = out / 'metrics' / 'benchmark_manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            mf = json.load(f)
        assert mf['config']['communication_range'] == 80.0, (
            f'CR not recorded: {mf["config"]}')


# ═══════════════════════════════════════════════════════════════════════════
# P0/P1 regression tests (setup-only isolation, area rejection, scanner, resume)
# ═══════════════════════════════════════════════════════════════════════════
def test_setup_only_area_damage_rejected():
    """P1-1: --setup-only --damage-mode area must exit with clear error."""
    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '10',
        '--terrains', 'ridge',
        '--seeds', '0',
        '--output-dir', '/tmp/_ugv_smoke_area_',
        '--setup-only',
        '--damage-mode', 'area',
    )
    assert rc != 0, f'expected non-zero exit, got {rc}'
    assert 'area' in stderr.lower() or 'supports --damage-mode random' in stderr, \
        f'expected area rejection message, got: {stderr}'


def test_setup_only_writes_isolated_csv(tmp_path):
    """P0-2: setup-only writes setup_scan.csv, not ugv_summary.csv."""
    out = tmp_path / 'p02'
    rc, stdout, _ = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '1',
        '--output-dir', str(out),
        '--setup-only',
    )
    setup_csv = out / 'metrics' / 'setup_scan.csv'
    benchmark_csv = out / 'metrics' / 'ugv_summary.csv'
    setup_json = out / 'metrics' / 'setup_scan.json'
    benchmark_json = out / 'metrics' / 'ugv_summary.json'

    assert setup_csv.exists(), (
        f'setup_scan.csv should exist\nstdout:\n{stdout}')
    assert not benchmark_csv.exists(), (
        'ugv_summary.csv must NOT exist for setup-only')
    assert setup_json.exists(), (
        f'setup_scan.json should exist')
    assert not benchmark_json.exists(), (
        'ugv_summary.json must NOT exist for setup-only')

    # Verify schema: no 'algorithm' column in setup CSV
    import csv
    with open(setup_csv, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
    assert 'algorithm' not in fieldnames, (
        f'setup_scan.csv must not have algorithm column, got {fieldnames}')
    assert 'K0' in fieldnames, f'Expected K0 column, got {fieldnames}'


def test_setup_only_resume(tmp_path):
    """P1-2: setup-only resume skips completed scenarios."""
    out = tmp_path / 'p12'
    # First run — use D=30 which reliably fragments ridge/N=50
    rc1, stdout1, _ = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '1',
        '--output-dir', str(out),
        '--setup-only',
    )
    # Second run with --resume should skip all
    rc2, stdout2, _ = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '1',
        '--output-dir', str(out),
        '--setup-only',
        '--resume',
    )
    assert 'skipping' in stdout2.lower(), (
        f'Expected resume skip message, got:\n{stdout2}')
    assert 'Resume: skipping' in stdout2, (
        f'Expected "Resume: skipping" in output')


def test_resume_fingerprint_mismatch_rejected(tmp_path):
    """P1-2: changing algorithms or horizon must cause fingerprint mismatch."""
    out = tmp_path / 'p12_fp'
    # First run
    _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
    )
    # Resume with different algorithms must fail
    rc, _, stderr = _run_bench(
        '--algorithms', 'CEN',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
        '--resume',
    )
    assert rc != 0, f'expected non-zero exit, got {rc}'
    assert 'fingerprint' in stderr.lower(), (
        f'expected fingerprint mismatch error, got: {stderr}')


def test_fingerprint_written_without_resume(tmp_path):
    """P0: fingerprint must be written even on non-resume runs."""
    out = tmp_path / 'fp_no_resume'
    rc, stdout, _ = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
    )
    fp_path = out / 'metrics' / 'setup_scan_resume_fingerprint.json'
    assert fp_path.exists(), (
        f'fingerprint should exist after non-resume run, found: '
        f'{list(out.rglob("*"))}')
    with open(fp_path) as f:
        fp = json.load(f)
    assert 'horizon' in fp, f'missing "horizon" in {fp}'
    assert 'omega_max' in fp, f'missing "omega_max" in {fp}'
    assert 'git_commit' in fp, f'missing "git_commit" in {fp}'


def test_resume_rejects_missing_fingerprint(tmp_path):
    """P0: --resume with CSV but no fingerprint must error."""
    out = tmp_path / 'fp_missing'
    out.mkdir(parents=True)
    # Manually create a CSV without fingerprint
    metrics = out / 'metrics'
    metrics.mkdir(parents=True)
    with open(metrics / 'setup_scan.csv', 'w') as f:
        f.write('terrain,D,seed_index,destroy_list_hash,K0,n_remain\n')

    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
        '--resume',
    )
    assert rc != 0, f'expected non-zero exit, got {rc}'
    assert 'fingerprint' in stderr.lower(), (
        f'expected fingerprint missing error, got: {stderr}')


def test_resume_all_skipped_returns_zero(tmp_path):
    """P1: when --resume skips all scenarios, exit code must be 0."""
    out = tmp_path / 'resume_zero'
    # First run (creates data + fingerprint)
    rc1, _, _ = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
    )
    # Second run with --resume (all already completed)
    rc2, stdout2, _ = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
        '--resume',
    )
    assert rc2 == 0, f'full resume should return 0, got {rc2}\nstdout:\n{stdout2}'
    assert 'nothing to do' in stdout2.lower() or 'already complete' in stdout2.lower(), (
        f'expected "nothing to do" message, got:\n{stdout2}')


def test_scanner_rejects_reversed_seed_range():
    """P2: --seeds 5 0 must be rejected."""
    rc, _, stderr = _run_scanner(
        '--crs', '120',
        '--D-values', '20',
        '--seeds', '5', '0',
        '--N', '50',
        '--terrains', 'ridge',
        '--output', '/tmp/_scanner_rev_test.json',
    )
    assert rc != 0, f'expected non-zero exit for reversed seed range, got {rc}'


def test_reuse_output_dir_without_resume_rejected(tmp_path):
    """P0: running without --resume on existing CSV must error."""
    out = tmp_path / 'reuse'
    # First run
    _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
    )
    # Second run without --resume — must fail
    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seeds', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
    )
    assert rc != 0, f'expected non-zero exit, got {rc}'
    assert 'Existing output found' in stderr, \
        f'expected existing-output error, got: {stderr}'


def test_overwrite_truncates_old_csv(tmp_path):
    """P0: --overwrite must truncate old CSV, not append."""
    out = tmp_path / 'overwrite_truncate'
    # First run with 2 seeds (via range)
    rc1, _, _ = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seed-range', '0', '1',
        '--output-dir', str(out),
        '--setup-only',
    )
    # Verify first run has rows
    import csv
    with open(out / 'metrics' / 'setup_scan.csv', newline='') as f:
        n_before = len(list(csv.DictReader(f)))
    assert n_before > 0, f'first run should have rows, got {n_before}'
    stale_invalid = out / 'metrics' / 'setup_invalid_cases.csv'
    stale_invalid.write_text('stale\n')
    stale_case = out / 'setup_cases' / 'stale.json'
    stale_case.parent.mkdir()
    stale_case.write_text('{}')

    # Second run with --overwrite, fewer seeds
    rc2, stdout, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seed-range', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
        '--overwrite',
    )
    assert rc2 == 0, f'expected zero exit with --overwrite, got {rc2}\nstderr:\n{stderr}'
    # Verify CSV was truncated (only 1 seed, not previous + new)
    with open(out / 'metrics' / 'setup_scan.csv', newline='') as f:
        n_after = len(list(csv.DictReader(f)))
    assert n_after == 1, (
        f'overwrite should truncate old CSV, expected 1 row, got {n_after}')
    assert not stale_invalid.exists(), 'overwrite must remove stale invalid cases'
    assert not stale_case.parent.exists(), 'overwrite must remove stale cases'


def test_stale_output_without_csv_requires_overwrite(tmp_path):
    """A stale mode-specific artifact must not be silently reused."""
    out = tmp_path / 'stale_artifact'
    metrics = out / 'metrics'
    metrics.mkdir(parents=True)
    (metrics / 'setup_invalid_cases.csv').write_text('stale\n')
    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '30',
        '--terrains', 'ridge',
        '--seed-range', '0', '0',
        '--output-dir', str(out),
        '--setup-only',
    )
    assert rc != 0, f'expected non-zero exit, got {rc}'
    assert 'Existing output found' in stderr, f'got: {stderr}'


def test_duplicate_algorithms_rejected():
    """P2: --algorithms MORPHICA MORPHICA must be rejected."""
    rc, _, stderr = _run_bench(
        '--algorithms', 'MORPHICA', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '10',
        '--terrains', 'flat',
        '--output-dir', '/tmp/_ugv_dup_algo_',
    )
    assert rc != 0, f'expected non-zero exit for duplicate algorithms, got {rc}'


def test_scanner_rejects_negative_seed():
    """P2: negative seed range must be rejected."""
    rc, _, stderr = _run_scanner(
        '--crs', '120',
        '--D-values', '20',
        '--seeds', '-1', '5',
        '--N', '50',
        '--terrains', 'ridge',
        '--output', '/tmp/_scanner_neg_test.json',
    )
    assert rc != 0, f'expected non-zero exit for negative seed, got {rc}'


def test_scanner_matches_setup_only(tmp_path):
    """P0-1: scanner must produce same results as --setup-only benchmark."""
    out = tmp_path / 'scanner_consistency'
    out.mkdir(parents=True)

    # -- Run scanner (seed=2, D=20, N=50, ridge: known K0>1) --
    scanner_rc, scanner_out, _ = _run_scanner(
        '--crs', '120',
        '--D-values', '20',
        '--seeds', '2', '2',
        '--N', '50',
        '--terrains', 'ridge',
        '--output', str(out / 'scenarios.json'),
    )
    assert scanner_rc == 0, f'scanner failed:\n{scanner_out}'

    with open(out / 'scenarios.json') as f:
        data = json.load(f)
    scanner_entries = data['results']['ridge']['120']
    assert len(scanner_entries) > 0, (
        f'scanner found no valid scenarios for seed=5')

    # -- Run setup-only benchmark on same case --
    bench_rc, bench_out, _ = _run_bench(
        '--algorithms', 'MORPHICA',
        '--num-agents', '50',
        '--destroyed-values', '20',
        '--terrains', 'ridge',
        '--seeds', '2',
        '--output-dir', str(out / 'bench'),
        '--setup-only',
    )
    # Read the setup_scan.csv produced by benchmark
    import csv
    bench_csv_n = out / 'bench' / 'metrics' / 'setup_scan.csv'
    assert bench_csv_n.exists(), (
        f'setup_scan.csv missing from benchmark\nstdout:\n{bench_out}')
    with open(bench_csv_n, newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f'expected 1 row, got {len(rows)}'

    # -- Verify consistency --
    bench_hash = rows[0]['destroy_list_hash']
    bench_K0 = int(rows[0]['K0'])

    import hashlib
    scanner_hash = hashlib.sha256(
        str(scanner_entries[0]['destroy_list']).encode()).hexdigest()[:12]
    scanner_K0 = scanner_entries[0]['K0']

    assert scanner_hash == bench_hash, (
        f'destroy_list_hash mismatch: scanner={scanner_hash}, '
        f'bench={bench_hash}')
    assert scanner_K0 == bench_K0, (
        f'K0 mismatch: scanner={scanner_K0}, bench={bench_K0}')


def _run_scanner(*args):
    """Run the scanner script and return (returncode, stdout, stderr)."""
    rc = subprocess.run(
        [sys.executable, 'ugv/scan_scenarios.py'] + list(args),
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    )
    return rc.returncode, rc.stdout, rc.stderr


# ═══════════════════════════════════════════════════════════════════════════
# runner
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import inspect

    tests = [(name, fn) for name, fn in inspect.getmembers(
        sys.modules[__name__], inspect.isfunction)
             if name.startswith('test_')]

    passed = 0
    for name, fn in tests:
        try:
            sig = inspect.signature(fn)
            if 'tmp_path' in sig.parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            passed += 1
            print(f'  PASS  {name}')
        except AssertionError as e:
            print(f'  FAIL  {name}: {e}')
        except Exception as e:
            print(f'  ERROR {name}: {e}')

    print(f'\n{passed}/{len(tests)} passed')
    sys.exit(0 if passed == len(tests) else 1)

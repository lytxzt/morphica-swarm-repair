"""Recover the original N=1000 ablation cohort without editing historical logs.

The old text exports numbered a case-offset-100 run locally from zero. Replay
all four variants on that entire pre-existing cohort, retaining the separately
recorded offset-zero cohort when building the paper statistics.
"""
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from run_control_suites import ROOT, FIELDS, section, worker, read_rows, validate_rows, verify_archive

ARCHIVED = ROOT / 'data/control_suites/ablation_cohort_b.csv'
OUT = ROOT / 'outputs/control_suites/ablation_cohort_b.csv'
MANIFEST = OUT.with_suffix('.manifest.json')
VARIANTS = ['full', 'no-R', 'no-P', 'no-S']
CASES = list(range(100, 150))


def sources():
    names = ['experiments/run_ablation_cohort_b.py',
             'experiments/run_control_suites.py', 'controllers/MORPHICA.py',
             'controllers/FiniteRelayMORPHICA.py',
             'Configurations.py', 'Utils.py',
             'data/configs/swarm_positions_1000.xlsx']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names}


def validate(rows):
    validate_rows(rows, [('ablation_cohort_b', 1000, 500, case, variant)
                         for case in CASES for variant in VARIANTS])
    expected = {(case, variant) for case in CASES for variant in VARIANTS}
    actual = {(int(r['case']), r['variant']) for r in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError('Incomplete or duplicate second cohort')
    for case in CASES:
        rs = [r for r in rows if int(r['case']) == case]
        mask = hashlib.sha256(json.dumps(sorted(
            random.Random(case + 1725).sample(range(1000), 500))).encode()).hexdigest()
        if any(r['mask_sha256'] != mask or int(r['seed']) != case + 1725 for r in rs):
            raise ValueError(f'Incorrect damage provenance: {case}')
        if len({int(r['initial_K']) for r in rs}) != 1:
            raise ValueError(f'Unmatched starting components: {case}')
        for r in rs:
            event = int(r['final_K']) == 1 and int(r['T_conn']) < 1799
            if bool(int(r['success'])) != event:
                raise ValueError(f'Inconsistent event: {case}/{r["variant"]}')
    comparisons = {}
    for variant, folder in [('no-P', 'MORPHICA_noP'), ('no-S', 'MORPHICA_noS')]:
        path = ROOT / f'data/logs/algorithm_sweep/N1000/{folder}/MORPHICA_d500_50cases.txt'
        historical = section(path, 'connect_step')
        replay = [int(next(r for r in rows if int(r['case']) == case
                           and r['variant'] == variant)['T_conn']) for case in CASES]
        comparisons[variant] = {
            'historical_source': str(path.relative_to(ROOT)),
            'historical_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'local_to_actual_case_offset': 100,
            'matching_outcomes': sum(a == b for a, b in zip(historical, replay)),
            'historical_count': len(historical),
            'replay_failures': [case for case, t in zip(CASES, replay) if t >= 1799],
        }
        if len(historical) != 50 or historical != replay:
            raise ValueError(f'Historical cohort regeneration failed: {variant}')
    return comparisons


def main():
    global OUT, MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=OUT.parent)
    parser.add_argument('--verify-archived', action='store_true')
    args = parser.parse_args()
    if args.verify_archived:
        rows = verify_archive(ARCHIVED, [('ablation_cohort_b', 1000, 500, case, variant)
                                        for case in CASES for variant in VARIANTS])
        comparisons = validate(rows)
        print(f'ablation_cohort_b: {len(rows)} archived rows verified', flush=True)
        print(json.dumps(comparisons), flush=True)
        return
    OUT = args.output_dir.resolve() / OUT.name
    MANIFEST = OUT.with_suffix('.manifest.json')
    if OUT == ARCHIVED.resolve():
        parser.error('Use a new output directory; data/control_suites contains archived results')
    hashes = sources()
    if OUT.exists() and not MANIFEST.exists():
        raise ValueError('Existing rows have no source manifest; use a new output directory')
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text())['sources'] != hashes:
            raise ValueError('Source changed; retain recorded cohort and use a new output')
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({
            'N': 1000, 'D': 500, 'cases': CASES, 'variants': VARIANTS,
            'damage_seed': 'actual case + 1725', 'sources': hashes,
            'selection': 'entire pre-existing historical cohort, fixed before replay',
            'retain_offset_zero_cohort': True,
        }, indent=2) + '\n')
    rows = read_rows(OUT) if OUT.exists() else []
    done = {(int(r['case']), r['variant']) for r in rows}
    if len(rows) != len(done):
        raise ValueError('Duplicate task')
    validate_rows(rows, [('ablation_cohort_b', 1000, 500, case, variant)
                         for case in CASES for variant in VARIANTS], complete=False)
    tasks = [('ablation', 1000, 500, case, variant)
             for case in CASES for variant in VARIANTS if (case, variant) not in done]
    if not tasks:
        comparisons = validate(rows)
        print(json.dumps(comparisons), flush=True)
        return
    with OUT.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not rows:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(worker, task): task for task in tasks}
            for future in as_completed(futures):
                row = future.result()
                row['suite'] = 'ablation_cohort_b'
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(f'{len(rows)}/200 case={row["case"]} {row["variant"]} '
                      f'T={row["T_conn"]} success={row["success"]}', flush=True)
    comparisons = validate(rows)
    metadata = json.loads(MANIFEST.read_text())
    metadata['historical_regeneration'] = comparisons
    metadata['rows'] = len(rows)
    metadata['csv_sha256'] = hashlib.sha256(OUT.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()

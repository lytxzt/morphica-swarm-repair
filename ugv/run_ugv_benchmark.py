#!/usr/bin/env python3
"""Multi-case UGV benchmark runner (Phase 2).

Runs algorithm × terrain × seed × D combinations with matched destruction
lists, computes source-free process metrics, and writes summary CSV + JSON.

Usage:
  python ugv/run_ugv_benchmark.py \
    --algorithms MORPHICA CEN HERO CR-MGC DEMD \
    --num-agents 50 --destroyed-values 20 \
    --terrains flat rolling \
    --seeds 0 1 2 3 4 \
    --output-dir data/logs/ugv
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ugv.run_ugv_case import run_case, prepare_scenario
from ugv.metrics import compute_process_metrics, compute_summary_row


@contextmanager
def _suppress_stdout():
    """Suppress algorithm stdout (e.g. DEMD per-episode training prints)."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def main():
    from Configurations import config_random_seed as _seed_pool
    _max_seed = len(_seed_pool) - 1

    _p = argparse.ArgumentParser()
    _p.add_argument('--algorithms', type=str, nargs='+', required=True,
                    help='Algorithm names, e.g. MORPHICA CEN HERO')
    _p.add_argument('--num-agents', type=int, default=None,
                    help='Override N (default: Configurations default)')
    _p.add_argument('--destroyed-values', type=int, nargs='+', default=[20],
                    help='D values to sweep')
    _p.add_argument('--terrains', type=str, nargs='+', required=True,
                    help='Terrain names, e.g. flat rolling')
    _seed_group = _p.add_mutually_exclusive_group()
    _seed_group.add_argument('--seeds', type=int, nargs='+', default=None,
                             help='Explicit seed list, e.g. --seeds 0 1 2 3 4')
    _seed_group.add_argument('--seed-range', type=int, nargs=2, default=None,
                             metavar=('FIRST', 'LAST'),
                             help='Seed range (inclusive), e.g. --seed-range 0 4')
    _p.add_argument('--output-dir', type=str, default='data/logs/ugv')
    _p.add_argument('--horizon', type=int, default=None)
    _p.add_argument('--omega-max', type=float, default=None)
    _p.add_argument('--communication-range', type=int, default=None,
                    help='Override CR (default: Configurations constant)')
    _p.add_argument('--damage-mode', type=str, default='random',
                    choices=['random', 'area'])
    _p.add_argument('--save-trajectories', action='store_true', default=False,
                    help='Save individual case JSON/NPZ files')
    _p.add_argument('--scenario-manifest', type=str, default=None,
                    help='Path to scan_scenarios.json — restricts runs to '
                         'pre-verified (seed, D) pairs.  Use with --crs to '
                         'compute intersection across multiple CR values.')
    _p.add_argument('--manifest-crs', type=int, nargs='+', default=None,
                    metavar='CR',
                    help='CR subset for scenario-manifest intersection '
                         '(default: all CRs in manifest)')
    _p.add_argument('--resume', action='store_true', default=False,
                    help='Skip scenarios already present in the output CSV '
                         '(reads --output-dir/metrics/ugv_summary.csv)')
    _p.add_argument('--overwrite', action='store_true', default=False,
                    help='Delete existing output CSV/JSON/fingerprint and '
                         'start fresh. Without --resume or --overwrite, '
                         'refusing to write to a directory that already '
                         'contains results.')
    _p.add_argument('--setup-only', action='store_true', default=False,
                    help='Only run init/destroy/reposition/K(0) check; '
                         'record valid seeds and exit without algorithm recovery')
    args = _p.parse_args()

    # -- P1: reject incompatible flag combinations --
    if args.resume and args.overwrite:
        _p.error('--resume and --overwrite are mutually exclusive. '
                 'Use --resume to continue, or --overwrite to start fresh.')
    if args.setup_only and args.damage_mode != 'random':
        _p.error('--setup-only only supports --damage-mode random. '
                 'Area damage requires Environment which is algorithm-dependent.')

    # --seeds / --seed-range resolution with validation
    if args.seed_range is not None:
        first, last = args.seed_range
        if first > last:
            _p.error(f'--seed-range: FIRST ({first}) must be <= LAST ({last})')
        if first < 0 or last > _max_seed:
            _p.error(f'--seed-range [{first}..{last}] out of bounds '
                     f'(valid: 0..{_max_seed})')
        seeds = list(range(first, last + 1))
    elif args.seeds is not None:
        for s in args.seeds:
            if s < 0 or s > _max_seed:
                _p.error(f'--seeds: {s} is out of bounds '
                         f'(valid: 0..{_max_seed})')
        seeds = args.seeds
    else:
        seeds = [0, 1, 2, 3, 4]  # default: 5 seeds

    if not seeds:
        _p.error('No seeds specified (empty list or range)')

    algorithms = args.algorithms
    D_values = args.destroyed_values
    terrains = args.terrains

    # -- reject duplicate algorithm names --
    if len(algorithms) != len(set(algorithms)):
        _p.error(f'Duplicate algorithm names in --algorithms: {algorithms}')

    # N resolution — must be early (used by resume fingerprint)
    from Configurations import config_num_of_agents as _default_N
    N = args.num_agents if args.num_agents is not None else _default_N

    # ── scenario manifest: restrict to pre-verified (seed, D) pairs ──
    scenario_filter = None  # set of (terrain, seed_idx, D)
    if args.scenario_manifest is not None:
        with open(args.scenario_manifest) as f:
            _sm = json.load(f)
        _sm_crs = args.manifest_crs if args.manifest_crs is not None \
                  else _sm['config']['CRs']
        print(f'Scenario manifest: {args.scenario_manifest}')
        print(f'  CR subset for intersection: {_sm_crs}')

        # Build per-terrain (seed, D) sets for each CR, then intersect
        for t in terrains:
            t_key = t  # terrain name as in manifest
            if t_key not in _sm['results']:
                _p.error(f'Terrain {t} not found in scenario manifest')
            cr_sets = []
            for cr in _sm_crs:
                s = set()
                for v in _sm['results'][t_key].get(str(cr), []):
                    s.add((v['seed_idx'], v['D']))
                if not s:
                    print(f'  WARNING: CR={cr} terrain={t} has 0 valid scenarios')
                cr_sets.append(s)
            if not cr_sets:
                _p.error(f'No valid scenarios for {t} at any requested CR')
            t_common = cr_sets[0].copy()
            for s in cr_sets[1:]:
                t_common &= s
            # Apply D filter (--destroyed-values) to manifest intersections
            _before_D = len(t_common)
            t_common = {(s, d) for s, d in t_common if d in D_values}
            if _before_D != len(t_common):
                print(f'  {t}: {len(t_common)} pairs after D-filter '
                      f'(D in {sorted(D_values)}) (was {_before_D})')
            print(f'  {t}: {len(t_common)} matched (seed,D) pairs after '
                  f'intersection across CRs {_sm_crs}')

            # Merge into global filter
            if scenario_filter is None:
                scenario_filter = set()
            for seed_idx, D in t_common:
                scenario_filter.add((t, seed_idx, D))

        if not scenario_filter:
            _p.error('Scenario manifest intersection is empty — '
                     'no (terrain, seed, D) pair valid at all requested CRs')
        print(f'  Total matched scenarios: {len(scenario_filter)}')

    # ── resolve git commit + dirty status + diff hash (fingerprint needs all) ──
    try:
        import hashlib as _hashlib
        import subprocess as _sp
        _git_rev = _sp.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=_PROJECT_ROOT).stdout.strip()
        _git_status = _sp.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, cwd=_PROJECT_ROOT).stdout.strip()
        _git_dirty = bool(_git_status)
        _git_diff_hash = ''
        if _git_dirty:
            _git_diff = _sp.run(
                ['git', 'diff', 'HEAD', '--binary'],
                capture_output=True, text=True, cwd=_PROJECT_ROOT).stdout
            _untracked_paths = _sp.run(
                ['git', 'ls-files', '--others', '--exclude-standard'],
                capture_output=True, text=True, cwd=_PROJECT_ROOT).stdout
            _untracked_hashes = ''
            if _untracked_paths:
                _untracked_hashes = _sp.run(
                    ['git', 'hash-object', '--stdin-paths'],
                    input=_untracked_paths, capture_output=True, text=True,
                    cwd=_PROJECT_ROOT).stdout
            _git_diff_hash = _hashlib.sha256(
                (_git_status + '\0' + _git_diff + '\0' +
                 _untracked_hashes).encode()).hexdigest()[:16]
    except Exception:
        _git_rev = 'unknown'
        _git_dirty = None
        _git_diff_hash = ''

    # Formal runs must be reproducible before they touch any output files.
    # `git status --porcelain` covers staged, unstaged, and untracked files.
    if not args.setup_only and _git_dirty:
        _p.error(
            'Working tree is dirty — formal benchmark requires a clean '
            'working tree for reproducible results.\n'
            'Commit or stash changes first, or use --setup-only for '
            'algorithm-free scanning.')

    # ── resolve output paths after all non-destructive validation ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = out_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── resolve horizon / omega_max defaults for fingerprint ──
    from Configurations import config_width as _fp_cw
    _fp_horizon = args.horizon if args.horizon is not None \
        else int(0.8 * _fp_cw)
    from ugv.ugv_config import UGV_OMEGA_MAX as _fp_om_def
    _fp_omega_max = args.omega_max if args.omega_max is not None \
        else _fp_om_def

    # ── resume: skip scenarios already in the output CSV ──
    _resume_skip = set()  # (terrain, D, seed_idx)
    _n_resumed = 0  # track for exit-code decision
    if args.resume:
        from Configurations import config_communication_range  # local import

        _resume_csv_path = metrics_dir / (
            'setup_scan.csv' if args.setup_only else 'ugv_summary.csv')
        _resume_rows = []  # preserved rows from complete scenarios
        _fingerprint_path = _resume_csv_path.parent / \
            (_resume_csv_path.stem + '_resume_fingerprint.json')
        _expected_fp = dict(
            algorithms=sorted(algorithms),
            communication_range=int(config_communication_range),
            N=N,
            damage_mode=args.damage_mode,
            setup_only=args.setup_only,
            horizon=_fp_horizon,
            omega_max=round(_fp_omega_max, 3),
            git_commit=_git_rev,
            git_dirty=_git_dirty,
            git_diff_hash=_git_diff_hash,
        )
        # -- fingerprint check: reject missing or mismatched --
        if _resume_csv_path.is_file():
            if not _fingerprint_path.is_file():
                _p.error(
                    f'CSV exists at {_resume_csv_path} but resume fingerprint '
                    f'is missing ({_fingerprint_path}).\n'
                    'This CSV was created by an older version without fingerprint '
                    'support.\nUse a fresh --output-dir or remove the old CSV.')
            with open(_fingerprint_path) as f:
                _saved_fp = json.load(f)
            _fp_mismatches = []
            for k, v in _expected_fp.items():
                sv = _saved_fp.get(k)
                if sv != v:
                    _fp_mismatches.append(
                        f'{k}: expected={v}, saved={sv}')
            if _fp_mismatches:
                _p.error(
                    f'Resume fingerprint mismatch with {_fingerprint_path}:\n  ' +
                    '\n  '.join(_fp_mismatches) +
                    '\nRefusing to resume — config has changed. '
                    'Use a fresh --output-dir or remove the old CSV.')
            # -- CSV exists + fingerprint OK: parse completed scenarios --
            with open(_resume_csv_path, newline='') as f:
                reader = csv.DictReader(f)
                _resume_fieldnames = reader.fieldnames
                _all_rows = list(reader)
                _per_scenario = {}  # (t,d,s) -> list of row dicts
                for row in _all_rows:
                    try:
                        t = row.get('terrain', '')
                        d = int(row.get('D', -1))
                        s = int(row.get('seed_index', -1))
                        if not (t and d >= 0 and s >= 0):
                            continue
                        key = (t, d, s)
                        _per_scenario.setdefault(key, []).append(row)
                    except (ValueError, KeyError):
                        continue
                if args.setup_only:
                    for key in _per_scenario:
                        _resume_skip.add(key)
                        _resume_rows.extend(_per_scenario[key])
                else:
                    n_algo = len(algorithms)
                    for key, key_rows in _per_scenario.items():
                        # Require exact row count + algorithm set match
                        if len(key_rows) != n_algo:
                            continue
                        algos = {r.get('algorithm', '') for r in key_rows}
                        if algos != set(algorithms):
                            continue
                        # All rows must share the same destroy_list_hash
                        hashes = {r.get('destroy_list_hash', '') for r in key_rows}
                        if len(hashes) != 1 or '' in hashes:
                            continue
                        _resume_skip.add(key)
                        _resume_rows.extend(key_rows)
                _n_resumed = len(_resume_skip)
                _partial = len(_per_scenario) - _n_resumed
                if _partial:
                    print(f'Resume: {_partial} partially-written scenario(s) '
                          f'will be re-run (incomplete algorithm set)')
                    # Rewrite clean CSV with only complete scenarios
                    with open(_resume_csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(
                            f, fieldnames=_resume_fieldnames,
                            extrasaction='ignore')
                        writer.writeheader()
                        writer.writerows(_resume_rows)
                        f.flush()
                        os.fsync(f.fileno())
            print(f'Resume: skipping {_n_resumed} already-completed '
                  f'scenarios from {_resume_csv_path}')
        else:
            print(f'Resume: no existing CSV at {_resume_csv_path}, starting fresh')

    if scenario_filter is not None:
        total = len(algorithms) * len(scenario_filter)
        print(f'Benchmark: {len(algorithms)} algorithms × '
              f'{len(scenario_filter)} matched scenarios '
              f'= {total} cases')
    else:
        if args.setup_only:
            total = len(terrains) * len(D_values) * len(seeds)
            print(f'Setup-only scan: {len(terrains)} terrains × '
                  f'{len(D_values)} D × {len(seeds)} seeds '
                  f'= {total} scenarios')
        else:
            total = len(algorithms) * len(terrains) * len(D_values) * len(seeds)
            print(f'Benchmark: {len(algorithms)} algorithms × '
                  f'{len(terrains)} terrains × {len(D_values)} D × {len(seeds)} seeds '
                  f'= {total} cases')
    print(f'N={N}  damage={args.damage_mode}  output={out_dir}\n')

    # ── write benchmark manifest (plan §15) ──
    # _git_rev and _git_dirty already computed above; compute status lines for manifest
    _git_status_lines = _git_status.split('\n') if _git_dirty else []

    from Configurations import (
        config_communication_range, config_constant_speed,
        config_central_point,
    )
    from ugv.ugv_config import (
        UGV_DT, UGV_V_MAX, UGV_OMEGA_MAX, UGV_AGENT_RADIUS, UGV_MAX_SLOPE_DEG,
        ALGO_MODE, ONE_SHOT_MODES,
    )
    import platform as _plat
    manifest = {
        'git_commit': _git_rev,
        'git_worktree_dirty': _git_dirty,
        'git_status': _git_status_lines if _git_dirty else [],
        'python_version': sys.version,
        'platform': _plat.platform(),
        'cli_args': {k: (sorted(v) if isinstance(v, (set, frozenset))
                        else v if isinstance(v, (int, float, bool, str, type(None)))
                        else str(v))
                     for k, v in vars(args).items()},
        'N': N,
        'algorithms': algorithms,
        'D_values': D_values,
        'terrains': terrains,
        'seeds': seeds,
        'config': {
            'communication_range': float(config_communication_range),
            'constant_speed': float(config_constant_speed),
            'central_point': [float(x) for x in config_central_point],
        },
        'safety_controller': {
            'dt': UGV_DT, 'v_max': UGV_V_MAX, 'omega_max': UGV_OMEGA_MAX,
            'agent_radius': UGV_AGENT_RADIUS,
            'max_slope_deg': UGV_MAX_SLOPE_DEG,
            'candidate_grid': '7yaw × 2speed',
            'scoring_weights': {'w_dir': 0.55, 'w_prog': 0.20,
                                'w_cost': 0.15, 'w_slope': 0.10},
            'impassable': '1-cell dilated obstacles + slope > 25°',
        },
        'algo_modes': ALGO_MODE,
        'one_shot_modes': sorted(ONE_SHOT_MODES),
        'communication_radius_semantics': 'CR-only Euclidean 3D distance',
    }
    _mf_path = out_dir / 'metrics' / 'benchmark_manifest.json'
    _mf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_mf_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Manifest: {_mf_path}')

    # -- CSV writer (defined before loop for incremental writes) --
    def _write_csv(path, row_list, append=False, fieldnames=None):
        """Write rows to CSV.  If fieldnames is given, use it; otherwise discover.
        Returns the ordered fieldnames used.  In append mode, no header is written."""
        if not row_list:
            return fieldnames
        if fieldnames is not None:
            ordered = fieldnames
        else:
            all_keys = []
            for r in row_list:
                for k in r:
                    if k not in all_keys:
                        all_keys.append(k)
            id_keys = ['algorithm', 'terrain', 'damage_mode', 'N', 'D',
                       'seed_index', 'seed_value', 'terrain_seed',
                       'scenario_id', 'destroy_list_hash',
                       'dt', 'v_max', 'omega_max', 'agent_radius', 'max_slope_deg',
                       'communication_range', 'connectivity_model']
            ordered = [k for k in id_keys if k in all_keys]
            ordered += [k for k in all_keys if k not in ordered]
        mode = 'a' if append else 'w'
        with open(path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered, extrasaction='ignore')
            if not append:
                writer.writeheader()
            writer.writerows(row_list)
            f.flush()
            os.fsync(f.fileno())
        return ordered

    rows = []
    invalid_rows = []
    n_done = 0
    n_skipped = 0
    t_start = time.time()
    result_stem = 'setup_scan' if args.setup_only else 'ugv_summary'
    csv_path = metrics_dir / f'{result_stem}.csv'
    json_path = metrics_dir / f'{result_stem}.json'
    invalid_csv_path = metrics_dir / (
        'setup_invalid_cases.csv' if args.setup_only else 'invalid_cases.csv')
    cases_dir = out_dir / ('setup_cases' if args.setup_only else 'cases')
    fingerprint_path = metrics_dir / f'{result_stem}_resume_fingerprint.json'
    _csv_fieldnames = None  # determined on first incremental write

    # -- handle existing output --
    _output_artifacts = [csv_path, json_path, invalid_csv_path,
                         fingerprint_path, cases_dir]
    if any(path.exists() for path in _output_artifacts):
        if args.resume:
            pass  # handled above in resume block
        elif args.overwrite:
            # Remove every mode-specific output, including trajectory files.
            for _p_rm in _output_artifacts:
                if _p_rm.is_dir():
                    shutil.rmtree(_p_rm)
                    print(f'  [overwrite] removed {_p_rm}')
                elif _p_rm.exists():
                    _p_rm.unlink()
                    print(f'  [overwrite] removed {_p_rm}')
        else:
            _p.error(
                f'Existing output found under {metrics_dir}\n'
                'Use --resume to continue from existing results, '
                'or --overwrite to discard and start fresh.')

    # -- write fingerprint BEFORE first CSV row (crash recovery) --
    _fp_path = fingerprint_path
    from Configurations import config_communication_range as _fp_cr, config_width as _fp_cw2
    from ugv.ugv_config import UGV_OMEGA_MAX as _fp_om
    _fp_h = args.horizon if args.horizon is not None else int(0.8 * _fp_cw2)
    _fp_w = args.omega_max if args.omega_max is not None else _fp_om
    _fp = dict(
        algorithms=sorted(algorithms),
        communication_range=int(_fp_cr),
        N=N,
        damage_mode=args.damage_mode,
        setup_only=args.setup_only,
        horizon=_fp_h,
        omega_max=round(_fp_w, 3),
        git_commit=_git_rev,
        git_dirty=_git_dirty,
        git_diff_hash=_git_diff_hash,
    )
    with open(_fp_path, 'w') as f:
        json.dump(_fp, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # Build iteration order.  When using a scenario manifest, the
    # (terrain, D, seed) triple comes from the pre-computed intersection;
    # D_values and seeds lists are replaced by the filtered pairs.
    if scenario_filter is not None:
        _scenario_pairs = sorted(scenario_filter)
        _terrains = sorted(set(t for t, s, d in _scenario_pairs))
        # Collect (D, seed) pairs per terrain for deterministic iteration
        _per_terrain = {t: sorted(set((d, s) for tt, s, d in _scenario_pairs if tt == t))
                        for t in _terrains}
    else:
        _terrains = terrains
        _per_terrain = {t: [(d, s) for d in D_values for s in seeds]
                        for t in terrains}

    for terrain_name in _terrains:
        for D, seed_idx in _per_terrain[terrain_name]:
                # -- resume skip: already in output CSV --
                if (terrain_name, D, seed_idx) in _resume_skip:
                    continue

                # -- setup-only mode: algorithm-free init scan --
                if args.setup_only:
                    try:
                        sc = prepare_scenario(
                            N, D, terrain_name, seed_idx,
                            damage_mode=args.damage_mode,
                            output_dir=(str(cases_dir)
                                        if args.save_trajectories else None))
                    except (ValueError, RuntimeError) as e:
                        print(f'  SKIP {terrain_name} D={D} seed={seed_idx}: '
                              f'init failed ({e})')
                        n_skipped += 1
                        continue

                    if not sc['valid_init']:
                        n_skipped += 1
                        # Log even invalid inits for completeness
                        dl_hash = hashlib.sha256(
                            str(sc.get('destroy_list', [])).encode()
                        ).hexdigest()[:12] if sc.get('destroy_list') else ''
                        invalid_row = {
                            'terrain': terrain_name, 'D': D,
                            'seed_index': seed_idx,
                            'K0': sc['K0'],
                            'n_remain': sc['n_remain'],
                            'init_repositioned': len(
                                sc.get('init_repositioned', [])),
                            'destroy_list_hash': dl_hash,
                        }
                        invalid_rows.append(invalid_row)
                        continue

                    n_done += 1
                    dl_hash = hashlib.sha256(
                        str(sc['destroy_list']).encode()).hexdigest()[:12]
                    setup_row = {
                        'terrain': terrain_name, 'D': D,
                        'seed_index': seed_idx,
                        'K0': sc['K0'],
                        'n_remain': sc['n_remain'],
                        'init_repositioned': len(
                            sc.get('init_repositioned', [])),
                        'destroy_list_hash': dl_hash,
                    }
                    rows.append(setup_row)
                    _write_csv(csv_path, [setup_row],
                               append=(_csv_fieldnames is not None),
                               fieldnames=(_csv_fieldnames or None))
                    if _csv_fieldnames is None:
                        _csv_fieldnames = ['terrain', 'D', 'seed_index', 'K0',
                                           'n_remain', 'init_repositioned',
                                           'destroy_list_hash']
                    elapsed = time.time() - t_start
                    rate = n_done / max(elapsed, 1.0)
                    print(f'  [{n_done}] {terrain_name} D={D} '
                          f'seed={seed_idx} K0={sc["K0"]} '
                          f'({rate:.1f}/s)', flush=True)
                    continue

                # -- first algorithm: generate destruction list --
                first_algo = algorithms[0]
                scenario_saved_paths = []
                try:
                    with _suppress_stdout():
                        meta, traj, summary, saved_paths = run_case(
                            algo_name=first_algo,
                            N=N, D=D,
                            terrain_name=terrain_name,
                            seed_idx=seed_idx,
                            horizon=args.horizon,
                            omega_max=args.omega_max,
                            damage_mode=args.damage_mode,
                            output_dir=(str(cases_dir)
                                        if args.save_trajectories else None),
                            save_output=args.save_trajectories,
                            verbose=False,
                        )
                    scenario_saved_paths.extend(saved_paths)
                except (ValueError, RuntimeError) as e:
                    print(f'  SKIP {terrain_name} D={D} seed={seed_idx}: '
                          f'{first_algo} setup failed ({e})')
                    n_skipped += len(algorithms)
                    continue

                destroy_list = meta.get('destroy_list', [])
                dl_hash = hashlib.sha256(
                    str(sorted(destroy_list)).encode()).hexdigest()[:12]
                scenario_id = (f'{terrain_name}_N{N}_D{len(destroy_list)}'
                               f'_r{D}_s{seed_idx}')

                if not destroy_list or summary.get('invalid_initial_graph'):
                    n_skipped += len(algorithms)
                    if meta:
                        pm = compute_process_metrics(
                            traj or [], summary.get('T_conn'))
                        row = compute_summary_row(meta, summary, pm)
                        row['scenario_id'] = scenario_id
                        row['destroy_list_hash'] = dl_hash
                        invalid_rows.append(row)
                    continue

                # record first algorithm (buffered — committed only if all pass)
                pm = compute_process_metrics(traj, summary.get('T_conn'))
                row = compute_summary_row(meta, summary, pm)
                row['scenario_id'] = scenario_id
                row['destroy_list_hash'] = dl_hash
                scenario_rows = [row]
                scenario_ok = True

                # -- remaining algorithms with matched destruction --
                for algo_name in algorithms[1:]:
                    try:
                        with _suppress_stdout():
                            meta2, traj2, summary2, saved_paths2 = run_case(
                                algo_name=algo_name,
                                N=N, D=D,
                                terrain_name=terrain_name,
                                seed_idx=seed_idx,
                                horizon=args.horizon,
                                omega_max=args.omega_max,
                                damage_mode=args.damage_mode,
                                output_dir=(str(cases_dir)
                                            if args.save_trajectories else None),
                                save_output=args.save_trajectories,
                                verbose=False,
                                destroy_list_override=destroy_list,
                            )
                        scenario_saved_paths.extend(saved_paths2)
                    except (ValueError, RuntimeError) as e:
                        print(f'  SKIP {terrain_name} D={D} seed={seed_idx} '
                              f'{algo_name}: {e}')
                        scenario_ok = False
                        break

                    pm2 = compute_process_metrics(
                        traj2, summary2.get('T_conn'))
                    row2 = compute_summary_row(meta2, summary2, pm2)
                    row2['scenario_id'] = scenario_id
                    row2['destroy_list_hash'] = dl_hash
                    scenario_rows.append(row2)

                # commit scenario only if all algorithms succeeded
                if scenario_ok:
                    rows.extend(scenario_rows)
                    n_done += len(scenario_rows)
                    # incremental CSV write — survive crashes (and resume)
                    if _csv_fieldnames is None:
                        if csv_path.is_file():
                            # resume: read existing header, append
                            with open(csv_path, newline='') as _f:
                                _csv_fieldnames = next(csv.reader(_f))
                            _write_csv(csv_path, scenario_rows, append=True,
                                       fieldnames=_csv_fieldnames)
                        else:
                            # fresh start: write with header
                            _csv_fieldnames = _write_csv(
                                csv_path, scenario_rows, append=False)
                    else:
                        _write_csv(csv_path, scenario_rows, append=True,
                                   fieldnames=_csv_fieldnames)
                else:
                    n_skipped += len(algorithms)  # entire scenario skipped
                    # clean up orphan trajectory files from failed scenario
                    for f in scenario_saved_paths:
                        if os.path.exists(f):
                            os.remove(f)

                # progress
                elapsed = time.time() - t_start
                rate = n_done / max(elapsed, 1.0)
                print(f'  [{n_done}/{total}] {terrain_name} D={D} '
                      f'seed={seed_idx}  ({rate:.1f} cases/s)',
                      end='\r' if n_done < total else '\n', flush=True)

    elapsed = time.time() - t_start
    print(f'\nDone: {n_done} cases, {n_skipped} skipped, '
          f'{elapsed:.1f}s total')

    # -- write invalid cases first (always, even if no valid rows) --
    if invalid_rows:
        _write_csv(invalid_csv_path, invalid_rows)
        print(f'Invalid cases: {invalid_csv_path}  ({len(invalid_rows)} rows)')

    # -- write resume fingerprint unconditionally (enables future --resume) --
    _fp_path = fingerprint_path
    from Configurations import config_communication_range as _fp_cr, config_width as _fp_cw2
    from ugv.ugv_config import UGV_OMEGA_MAX as _fp_om
    _fp_h = args.horizon if args.horizon is not None else int(0.8 * _fp_cw2)
    _fp_w = args.omega_max if args.omega_max is not None else _fp_om
    _fp = dict(
        algorithms=sorted(algorithms),
        communication_range=int(_fp_cr),
        N=N,
        damage_mode=args.damage_mode,
        setup_only=args.setup_only,
        horizon=_fp_h,
        omega_max=round(_fp_w, 3),
        git_commit=_git_rev,
        git_dirty=_git_dirty,
        git_diff_hash=_git_diff_hash,
    )
    with open(_fp_path, 'w') as f:
        json.dump(_fp, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    if not rows:
        # Distinguish "all skipped by resume" (success) from genuine failure
        if _n_resumed > 0:
            print(f'All {_n_resumed} scenario(s) already complete — nothing to do.')
            return 0
        print('No cases completed.')
        return 1

    # CSV was written incrementally; only report path
    print(f'CSV: {csv_path}  ({len(rows)} rows)')

    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f'JSON: {json_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

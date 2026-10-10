#!/usr/bin/env python3
"""Reproducible control experiments; never overwrite historical sweep logs.

Use the repository Python environment, e.g.:
  OPENBLAS_NUM_THREADS=1 python experiments/run_control_suites.py --suite null --workers 8
Suites: null (300 matched N=1000 random-waypoint trials), ablation (100
matched no-P/no-S reruns), nonideal (750 corrected observation trials), relay
(450 N=200 finite-flooding trials), symmetry (exact equilibrium diagnostic).
Logging uses the historical zero-based step index; failures are censored at
H-1, the common recording cutoff. Motion itself is in full three dimensions.
"""

from __future__ import annotations

import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARCHIVED = ROOT / 'data' / 'control_suites'
OUT = ROOT / 'outputs' / 'control_suites'
FIELDS = ['suite', 'N', 'D', 'case', 'variant', 'seed', 'mask_sha256',
          'horizon', 'initial_K', 'T_conn', 'success', 'final_K', 'seconds']


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def validate_rows(rows, tasks, complete=True):
    expected = set(tasks)
    actual = [(r['suite'], int(r['N']), int(r['D']), int(r['case']), r['variant'])
              for r in rows]
    if len(actual) != len(set(actual)):
        raise ValueError('Duplicate task')
    if not set(actual) <= expected or (complete and set(actual) != expected):
        raise ValueError('Recorded tasks do not match the suite')
    for r in rows:
        n, d, case = int(r['N']), int(r['D']), int(r['case'])
        seed = case + 1725
        mask = hashlib.sha256(json.dumps(sorted(
            random.Random(seed).sample(range(n), d))).encode()).hexdigest()
        if int(r['seed']) != seed or r['mask_sha256'] != mask:
            raise ValueError(f'Incorrect damage mask: {case}/{r["variant"]}')
        horizon = int(0.8 * {200: 1000, 1000: 2250}[n])
        t, k = int(r['T_conn']), int(r['final_K'])
        if int(r['horizon']) != horizon or not 0 <= t < horizon or k < 1:
            raise ValueError('Invalid horizon or terminal state')
        if int(r['initial_K']) <= 1 or int(r['success']) != int(k == 1 and t < horizon - 1):
            raise ValueError('Inconsistent connectivity event')


def verify_archive(path, tasks):
    rows = read_rows(path)
    metadata = json.loads(path.with_suffix('.manifest.json').read_text())
    validate_rows(rows, tasks)
    for name, digest in metadata['sources'].items():
        if 'swarm_positions_' in name:
            geometry = ROOT / 'data/configs' / Path(name).name
            if hashlib.sha256(geometry.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Archived geometry mismatch: {name}')
    if 'csv_sha256' in metadata:
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['csv_sha256']:
            raise ValueError('Archived CSV checksum mismatch')
    return rows


def section(path, name):
    match = re.search(re.escape(name) + r':\s*\n\s*(\[[^\]]*\])', path.read_text(), re.S)
    if match is None:
        raise ValueError(f'Missing {name}: {path}')
    return ast.literal_eval(match.group(1))


def labels(positions, cr=120.0):
    pairs = cKDTree(positions).query_pairs(cr, output_type='ndarray')
    n = len(positions)
    if len(pairs):
        ii = np.concatenate((pairs[:, 0], pairs[:, 1]))
        jj = np.concatenate((pairs[:, 1], pairs[:, 0]))
        graph = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n)).tocsr()
    else:
        graph = coo_matrix((n, n)).tocsr()
    k, lab = connected_components(graph, directed=False)
    return int(k), lab


def configure(n):
    # Configurations parses argv at import time; pass only the intended scale.
    sys.argv = [sys.argv[0], '--num-agents', str(n)]
    import Configurations as C
    return C


def worker(task):
    suite, n, d, case, variant = task
    C = configure(n)
    from controllers.MORPHICA import MORPHICA
    from controllers.FiniteRelayMORPHICA import FiniteRelayMORPHICA
    seed = case + 1725  # range(7,10007)[case] + 1718 in the original runner
    destroyed = random.Random(seed).sample(range(n), d)
    remain = [i for i in range(n) if i not in set(destroyed)]
    positions = C.config_initial_swarm_positions.copy()
    initial_k, _ = labels(positions[remain])
    if initial_k <= 1:
        raise ValueError(f'Matched case is not fragmented: {task}')
    horizon = int(0.8 * C.config_width)
    C._obs_packet_loss = C._obs_delay_steps = C._obs_position_noise = 0
    if suite == 'nonideal':
        specs = {'loss005': (.05, 0, 0), 'loss010': (.10, 0, 0),
                 'loss020': (.20, 0, 0), 'delay1': (0, 1, 0),
                 'delay3': (0, 3, 0), 'delay5': (0, 5, 0),
                 'noise2': (0, 0, 2), 'noise5': (0, 0, 5),
                 'noise10': (0, 0, 10), 'combined': (.10, 3, 5)}
        C._obs_packet_loss, C._obs_delay_steps, C._obs_position_noise = specs[variant]
    if suite == 'null':
        rng = np.random.default_rng(seed + 100000)
        waypoints = rng.uniform(0, C.config_space_range, size=(len(remain), 3))
        controller = None
    elif suite == 'relay' and variant != 'infinite':
        controller = FiniteRelayMORPHICA(positions, relay_rounds=int(variant), observation_seed=seed)
    else:
        controller = MORPHICA(positions, ablation=variant if suite == 'ablation' else 'full', observation_seed=seed)
    started = time.monotonic()
    k = initial_k
    for step in range(horizon):
        if suite == 'null':
            current = positions[remain]
            reached = np.linalg.norm(waypoints - current, axis=1) <= 1.0
            waypoints[reached] = rng.uniform(0, C.config_space_range, size=(int(reached.sum()), 3))
            delta = waypoints - current
            actions = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1.0)
            positions[remain] += actions
        else:
            actions, _ = controller.group_fusion_step(positions, remain)
            positions += actions * C.config_constant_speed
        k, _ = labels(positions[remain])
        if k == 1:
            break
    return dict(zip(FIELDS, [suite, n, d, case, variant, seed,
        hashlib.sha256(json.dumps(sorted(destroyed)).encode()).hexdigest(),
        horizon, initial_k, step, int(k == 1 and step < horizon - 1), k,
        round(time.monotonic() - started, 5)]))


def tasks_for(suite):
    if suite == 'null':
        configs = [(1000, d, ['random-waypoint'], 50) for d in [250, 350, 450, 550, 650, 750]]
    elif suite == 'ablation':
        configs = [(1000, 500, ['no-P', 'no-S'], 50)]
    elif suite == 'nonideal':
        configs = [(1000, d, ['loss005', 'loss010', 'loss020', 'delay1', 'delay3', 'delay5',
                              'noise2', 'noise5', 'noise10', 'combined'], 25) for d in [250, 500, 750]]
    elif suite == 'relay':
        configs = [(200, d, ['1', '2', '4', '8', '16', 'infinite'], 25) for d in [50, 100, 150]]
    else:
        raise ValueError(suite)
    tasks = []
    for n, d, variants, count in configs:
        path = ROOT / f'data/logs/algorithm_sweep/N{n}/MORPHICA/MORPHICA_d{d}_50cases.txt'
        cases = section(path, 'case')[:count]
        if len(cases) != count:
            raise ValueError(f'Insufficient matched cases: {path}')
        for variant in variants:
            tasks.extend((suite, n, d, case, variant) for case in cases)
    return tasks


def symmetry():
    C = configure(200)
    from controllers.MORPHICA import MORPHICA
    anchor = C.config_central_point.copy()
    angles = 2 * np.pi * np.arange(8) / 8
    positions = np.tile(anchor, (9, 1))
    positions[1:, :2] += 150 * np.column_stack((np.cos(angles), np.sin(angles)))
    outputs = {}
    for variant in ['no-P', 'full']:
        controller = MORPHICA(positions, anchor_region=anchor, ablation=variant)
        current = positions.copy()
        speed, _ = controller.group_fusion_step(current, list(range(9)))
        initial_speed = float(np.linalg.norm(speed, axis=1).max())
        final_k = labels(current)[0]
        final_step = None
        for step in range(2000):
            actions, _ = controller.group_fusion_step(current, list(range(9)))
            current += actions
            final_k = labels(current)[0]
            if final_k == 1:
                final_step = step + 1
                break
        outputs[variant] = {'initial_max_speed': initial_speed, 'K_final': final_k,
                            'steps': 2000 if final_step is None else final_step,
                            'max_displacement': float(np.linalg.norm(current - positions, axis=1).max())}
    assert outputs['no-P']['initial_max_speed'] < 1e-9
    assert outputs['no-P']['max_displacement'] < 1e-9
    assert outputs['no-P']['K_final'] == 2
    assert outputs['full']['initial_max_speed'] > .9
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'symmetry.json').write_text(json.dumps({'radius': 150, 'ring_agents': 8,
        'centre_agents': 1, 'CR': 120, 'margin': 20, 'results': outputs}, indent=2) + '\n')
    print(json.dumps(outputs), flush=True)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['null', 'ablation', 'nonideal', 'relay', 'symmetry'], required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='new run directory (default: outputs/control_suites)')
    parser.add_argument('--verify-archived', action='store_true',
                        help='validate recorded rows without rerunning or changing them')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if args.verify_archived:
        if args.suite == 'symmetry':
            parser.error('--verify-archived supports null, ablation, nonideal and relay')
        rows = verify_archive(ARCHIVED / f'{args.suite}.csv', tasks_for(args.suite))
        print(f'{args.suite}: {len(rows)} archived rows verified', flush=True)
        return
    if OUT == ARCHIVED.resolve():
        parser.error('Use a new output directory; data/control_suites contains archived results')
    if args.suite == 'symmetry':
        symmetry()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{args.suite}.csv'
    all_tasks = tasks_for(args.suite)
    done = set()
    if path.exists():
        rows = read_rows(path)
        validate_rows(rows, all_tasks, complete=False)
        for row in rows:
            done.add((row['suite'], int(row['N']), int(row['D']), int(row['case']), row['variant']))
    tasks = [t for t in all_tasks if t not in done]
    print(f'{args.suite}: {len(tasks)} remaining; {len(done)} recorded', flush=True)
    metadata = {'suite': args.suite, 'matched_task_count': len(all_tasks),
                'python': sys.version, 'workers': args.workers, 'cutoff': 'horizon - 1',
                'damage_seed': 'case + 1725', 'observation_seed': 'case + 1725',
                'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__).resolve(), ROOT / 'controllers/MORPHICA.py',
                                      ROOT / 'controllers/FiniteRelayMORPHICA.py',
                                      ROOT / 'Configurations.py', ROOT / 'Utils.py',
                                      ROOT / 'data/configs/swarm_positions_1000.xlsx',
                                      ROOT / 'data/configs/swarm_positions_200.xlsx']}}
    manifest = OUT / f'{args.suite}.manifest.json'
    if done and not manifest.exists():
        raise ValueError('Existing rows have no source manifest; use a new output directory')
    if manifest.exists() and json.loads(manifest.read_text())['sources'] != metadata['sources']:
        raise ValueError('Refusing to mix runs from different code or geometry')
    if not tasks and done:
        return
    manifest.write_text(json.dumps(metadata, indent=2) + '\n')
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        # One scale per suite makes fork-inherited Configurations unambiguous.
        configure(200 if args.suite == 'relay' else 1000)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(worker, tasks, chunksize=1), 1):
                writer.writerow(result)
                handle.flush()
                if i % 10 == 0 or i == len(tasks):
                    print(f'{args.suite}: {len(done) + i}/{len(done) + len(tasks)}', flush=True)


if __name__ == '__main__':
    main()

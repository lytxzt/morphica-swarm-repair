#!/usr/bin/env python3
"""Pre-compute (CR, D, seed) combinations where K(0) > 1.

Uses prepare_scenario() from run_ugv_case — identical init flow to the
actual benchmark.  For each CR, recomputes K(0) via compute_graph_state
with the explicit communication_range parameter.

Usage:
  python UGV/scan_scenarios.py \
    --crs 60 80 100 120 140 160 180 \
    --D-values 30 50 \
    --seeds 0 99 \
    --N 100 \
    --terrains ridge \
    --output Logs/ugv/cr_scenarios.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crs', type=int, nargs='+', required=True)
    ap.add_argument('--D-values', type=int, nargs='+', default=[30, 50],
                    metavar='D', help='Destruction levels to scan (default: 30 50)')
    ap.add_argument('--seeds', type=int, nargs=2, default=[0, 99],
                    metavar=('FIRST', 'LAST'))
    ap.add_argument('--terrains', type=str, nargs='+', default=['ridge'])
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--output', type=str, default='Logs/ugv/cr_scenarios.json')
    args = ap.parse_args()

    # Inject --num-agents into sys.argv so Configurations.py (which parses
    # sys.argv at import time) sees the correct N.  Must happen BEFORE the
    # first import from UGV or Configurations.
    if '--num-agents' not in sys.argv and '--num_agents' not in sys.argv:
        sys.argv.extend(['--num-agents', str(args.N)])

    # Late import — Configurations module is imported via this chain and
    # will pick up --num-agents from sys.argv.
    from UGV.run_ugv_case import prepare_scenario
    from UGV.ugv_environment import compute_graph_state

    N = args.N
    crs = sorted(args.crs)
    terrains = args.terrains
    D_vals = args.D_values

    from Configurations import config_random_seed as _seed_pool
    max_seed_idx = len(_seed_pool) - 1
    if args.seeds[0] > args.seeds[1]:
        ap.error(f'First seed ({args.seeds[0]}) must be <= last seed ({args.seeds[1]})')
    if args.seeds[0] < 0:
        ap.error(f'First seed must be >= 0, got {args.seeds[0]}')
    if args.seeds[1] > max_seed_idx:
        ap.error(f'Last seed must be <= {max_seed_idx}, got {args.seeds[1]}')
    seeds = list(range(args.seeds[0], args.seeds[1] + 1))

    results = {t: {str(cr): [] for cr in crs} for t in terrains}
    total_checked = 0
    total_reposition_fail = 0
    total_init_fail = 0

    for terrain_name in terrains:
        print(f'Terrain: {terrain_name}  N={N}')
        print(f'Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)})')
        print(f'D values: {D_vals}')
        print(f'CRs: {crs}')
        print()

        n_terrain_checked = 0
        for seed_idx in seeds:
            for D in D_vals:
                n_terrain_checked += 1
                total_checked += 1

                # -- algorithm-free scenario init --
                try:
                    sc = prepare_scenario(N, D, terrain_name, seed_idx,
                                          damage_mode='random')
                except RuntimeError as e:
                    # Agent repositioning failure — skip this seed entirely
                    total_reposition_fail += 1
                    if 'could not be repositioned' in str(e):
                        break  # this seed is broken for all D
                    continue
                except ValueError:
                    total_init_fail += 1
                    continue

                positions = sc['positions']
                remain = sc['remain']

                if len(remain) < 2:
                    continue

                # -- recompute K(0) at each CR --
                for cr in crs:
                    nc, _, _ = compute_graph_state(
                        positions, remain, communication_range=cr)
                    if nc > 1:
                        results[terrain_name][str(cr)].append({
                            'seed_idx': seed_idx,
                            'D': D,
                            'K0': nc,
                            'n_remain': len(remain),
                            'init_repositioned': len(
                                sc.get('init_repositioned', [])),
                            'destroy_list': sc.get('destroy_list', []),
                        })

            # progress per 10 seeds
            if (seed_idx - seeds[0] + 1) % 10 == 0:
                _done = seed_idx - seeds[0] + 1
                print(f'  seeds {seeds[0]}..{seed_idx} '
                      f'({_done}/{len(seeds)})', flush=True)

        # -- per-terrain summary --
        for cr in crs:
            n_valid = len(results[terrain_name][str(cr)])
            print(f'  CR={cr:>3d}: {n_valid:>4d} valid scenarios '
                  f'({100 * n_valid / max(n_terrain_checked, 1):.1f}%)',
                  flush=True)
        print()

    # ── Cross-CR intersection ──
    print('=== Cross-CR intersection ===')
    for terrain_name in terrains:
        sets = {}
        for cr in crs:
            s = set()
            for v in results[terrain_name][str(cr)]:
                s.add((v['seed_idx'], v['D']))
            sets[cr] = s
            print(f'  {terrain_name} CR={cr}: {len(s)} unique (seed,D) pairs')

        common = sets[crs[0]].copy() if crs else set()
        for cr in crs[1:]:
            common &= sets[cr]

        print(f'  Intersection ALL CRs: {len(common)} pairs')
        if common:
            by_D = {}
            for seed, D in sorted(common):
                by_D.setdefault(D, []).append(seed)
            for D in sorted(by_D):
                print(f'    D={D}: {len(by_D[D])} seeds')

        # Find largest CR subset with >= 15 scenarios
        print()
        print('=== CR subsets with stable scenario counts ===')
        for start in range(len(crs)):
            for end in range(len(crs) - 1, start, -1):
                subset = crs[start:end + 1]
                sub_common = sets[subset[0]].copy()
                for cr in subset[1:]:
                    sub_common &= sets[cr]
                if len(sub_common) >= 15:
                    print(f'  CR={subset[0]}–{subset[-1]}: '
                          f'{len(sub_common)} matched pairs '
                          f'(across {len(subset)} CRs)')
                    break

    # ── save ──
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'config': {
                'N': N, 'CRs': crs, 'D_values': D_vals,
                'seed_range': list(args.seeds), 'terrains': terrains,
            },
            'stats': {
                'n_checked': total_checked,
                'n_reposition_fail': total_reposition_fail,
                'n_init_fail': total_init_fail,
            },
            'results': results,
        }, f, indent=2)
    print(f'\nSaved: {out}')
    print(f'Stats: {total_checked} checked, '
          f'{total_reposition_fail} reposition failures, '
          f'{total_init_fail} init failures')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run a single UGV terrain recovery case.

Configurations.py handles --num-agents, --destroyed, --algorithm, --ablation
and --seed via parse_known_args at import time.  This script adds UGV-specific
flags (--terrain, --case-index, --output-dir, --omega-max, --horizon,
--damage-mode) via its own parse_known_args call.

Usage:
  python ugv/run_ugv_case.py \
    --num-agents 50 --destroyed 20 --algorithm MORPHICA \
    --terrain flat --case-index 0 --output-dir data/logs/ugv/cases
"""

import argparse
import json
import os
import sys
import time
import random
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path (ugv/ is a sub-package)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Configurations import *
from Environment import Environment
from Swarm import Swarm
import Utils

from ugv.terrain import TerrainMap
from ugv.ugv_config import (
    UGV_DT, UGV_V_MAX, UGV_OMEGA_MAX, UGV_AGENT_RADIUS,
    UGV_MAX_SLOPE_DEG, EPS,
    ALGO_MODE, ONE_SHOT_MODES,
    WAYPOINT_ARRIVE_THRESH, WAYPOINT_STOP_THRESH,
)
from ugv.ugv_environment import (
    ugv_step, compute_graph_state, init_yaw, footprint_hits_obstacle,
    _footprint_max_slope, _inside_shrunk_bounds,
)


# ═══════════════════════════════════════════════════════════════════════════
# one-shot waypoint actions  (§5)
# ═══════════════════════════════════════════════════════════════════════════
def _compute_waypoint_actions(positions, waypoints, remain_list):
    """Compute unit-horizontal actions steering each agent toward its waypoint.

    d > WAYPOINT_ARRIVE_THRESH  → unit vector toward waypoint
    0.01 < d <= 1.0             → direct vector (fine positioning)
    d <= 0.01                   → zero action (arrived)
    """
    N = len(positions)
    actions = np.zeros((N, 3), dtype=np.float64)
    for i in remain_list:
        d_xy = waypoints[i, :2] - positions[i, :2]
        dist = float(np.linalg.norm(d_xy))
        if dist > WAYPOINT_ARRIVE_THRESH:
            actions[i, :2] = d_xy / dist
        elif dist > WAYPOINT_STOP_THRESH:
            actions[i, :2] = d_xy          # fine positioning
        # else: stay at zero
    return actions


# ═══════════════════════════════════════════════════════════════════════════
# UGV solve loop
# ═══════════════════════════════════════════════════════════════════════════
def solve_ugv(swarm, terrain, horizon, algo_mode,
              dt=UGV_DT, v_max=UGV_V_MAX, omega_max=UGV_OMEGA_MAX,
              record_trajectory=True, verbose=True):
    """Run UGV solve loop until first K(t)=1 or horizon.

    Returns (step, final_positions, num_components, trajectory, summary).
    """
    N = swarm.num_of_agents
    remain_list = list(swarm.remain_list)

    positions = swarm.true_positions.copy()

    # positions are already terrain-projected before Swarm construction,
    # but re-project z for destroyed agents (may have stale z)
    for i in range(N):
        positions[i, 2] = terrain.height(positions[i, :2])
    swarm.update_true_positions(positions)
    swarm.make_remain_positions()

    # initial yaw toward formation centre (plan §9 step 4)
    yaw = init_yaw(positions)

    # per-step diagnostics accumulation
    blocked_steps = 0
    collision_steps = 0
    saturated_steps = 0
    path_length = np.zeros(N, dtype=np.float64)   # cumulative per-agent (m)
    terrain_cost = np.zeros(N, dtype=np.float64)  # ∫ cost × ds
    excess_terrain_cost = np.zeros(N, dtype=np.float64)  # ∫ (cost-1) × ds

    is_one_shot = algo_mode in ONE_SHOT_MODES
    waypoints = None  # set after first take_actions for one-shot modes

    # -- t=0 snapshot + connectivity check --
    nc0, cl0, ed0 = compute_graph_state(positions, remain_list)
    trajectory = []
    if record_trajectory:
        trajectory.append(dict(
            step=0, positions=positions.copy(), yaw=yaw.copy(),
            remain_list=list(remain_list), edges=ed0,
            component_id=cl0.tolist(),
            diagnostics={
                'blocked': [], 'detour_ratio': [], 'selected_score': [],
                'collision': [], 'raw_action': [], 'executed_action': [],
            },
        ))

    if nc0 == 1:
        summary = dict(
            success=False, T_conn=None,
            invalid_initial_graph=True,
            num_components_final=1, num_components_initial=1,
            horizon=horizon, n_remain=len(remain_list), n_total=N,
            blocked_steps=0, collision_steps=0,
            saturated_steps=0, control_saturation_ratio=0.0,
            path_length_total=0.0, path_length_mean=0.0,
            terrain_cost_total=0.0, terrain_cost_mean=0.0,
            excess_terrain_cost_total=0.0, excess_terrain_cost_mean=0.0,
            mean_cost_per_meter=1.0,
        )
        return 0, positions, 1, (trajectory if record_trajectory else None), summary

    step = 0
    for step in range(1, horizon + 1):
        # -- compute actions --
        if is_one_shot:
            if waypoints is None:
                swarm.take_actions()
                waypoints = positions.copy()
                waypoints[remain_list] = swarm.best_final_positions
                for i in remain_list:
                    waypoints[i, 2] = terrain.height(waypoints[i, :2])
            actions = _compute_waypoint_actions(
                positions, waypoints, remain_list)
        else:
            actions, _ = swarm.take_actions()

        # -- UGV dynamics with shared safety controller --
        pos_old = positions.copy()
        positions, yaw, diag = ugv_step(
            positions, actions, yaw, remain_list, terrain,
            dt=dt, v_max=v_max, omega_max=omega_max,
        )
        swarm.update_true_positions(positions)
        swarm.make_remain_positions()

        blocked_steps += int(np.sum(diag['blocked']))
        collision_steps += int(np.sum(diag['collision']))
        saturated_steps += int(np.sum(
            diag['saturated_yaw'] | diag['saturated_speed']))

        # per-step path metrics
        step_dist = np.linalg.norm(
            positions[remain_list, :2] - pos_old[remain_list, :2], axis=1)
        for idx, ri in enumerate(remain_list):
            path_length[ri] += float(step_dist[idx])
            mid_xy = 0.5 * (positions[ri, :2] + pos_old[ri, :2])
            c = float(terrain.cost(mid_xy))
            terrain_cost[ri] += float(step_dist[idx] * c)
            excess_terrain_cost[ri] += float(step_dist[idx] * (c - 1.0))

        nc, cl, ed = compute_graph_state(positions, remain_list)

        if record_trajectory:
            trajectory.append(dict(
                step=step, positions=positions.copy(), yaw=yaw.copy(),
                remain_list=list(remain_list), edges=ed,
                component_id=cl.tolist(),
                diagnostics={
                    'blocked': diag['blocked'].tolist(),
                    'detour_ratio': diag['detour_ratio'].tolist(),
                    'selected_score': diag['selected_score'].tolist(),
                    'collision': diag['collision'].tolist(),
                    'raw_action': diag['raw_action'].tolist(),
                    'executed_action': diag['executed_action'].tolist(),
                },
            ))

        if verbose:
            print(f"\r  step {step:4d}  subnets={nc:3d}  "
                  f"remain={len(remain_list):4d}",
                  end='', flush=True)

        if nc == 1:
            break

    if verbose:
        print()

    success = (nc == 1)
    remain_arr = np.array(remain_list, dtype=int)
    pl_sum = float(np.sum(path_length[remain_arr]))
    tc_sum = float(np.sum(terrain_cost[remain_arr]))
    ec_sum = float(np.sum(excess_terrain_cost[remain_arr]))
    summary = dict(
        success=success,
        T_conn=(step if success else None),
        num_components_final=nc,
        num_components_initial=nc0,
        horizon=horizon,
        n_remain=len(remain_list),
        n_total=N,
        blocked_steps=blocked_steps,
        collision_steps=collision_steps,
        saturated_steps=saturated_steps,
        control_saturation_ratio=float(
            saturated_steps / max(len(remain_list) * step, 1)),
        path_length_total=pl_sum,
        path_length_mean=float(pl_sum / max(len(remain_arr), 1)),
        terrain_cost_total=tc_sum,
        terrain_cost_mean=float(tc_sum / max(len(remain_arr), 1)),
        excess_terrain_cost_total=ec_sum,
        excess_terrain_cost_mean=float(ec_sum / max(len(remain_arr), 1)),
        mean_cost_per_meter=float(tc_sum / max(pl_sum, EPS)),
    )
    if is_one_shot:
        summary['planner_mode'] = 'waypoint_adapter'
        summary['replan_steps'] = 1
    return step, positions, nc, (trajectory if record_trajectory else None), summary


# ═══════════════════════════════════════════════════════════════════════════
# trajectory export
# ═══════════════════════════════════════════════════════════════════════════
def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_trajectory(trajectory, metadata, output_dir, prefix, verbose=True):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    jp = out / f'{prefix}.json'
    blob = _jsonable({'metadata': metadata, 'frames': trajectory})
    with open(jp, 'w') as f:
        json.dump(blob, f, indent=2)

    npz = out / f'{prefix}.npz'
    arrays = {}
    for fr in trajectory:
        s = fr['step']
        arrays[f'pos_{s}'] = np.asarray(fr['positions'])
        arrays[f'yaw_{s}'] = np.asarray(fr['yaw'])
        arrays[f'comp_{s}'] = np.asarray(fr['component_id'])
    np.savez_compressed(npz, **arrays)

    if verbose:
        print(f'Trajectory saved: {jp}  ({jp.stat().st_size} B)')
        print(f'                  {npz}  ({npz.stat().st_size} B)')
    return str(jp), str(npz)


# ═══════════════════════════════════════════════════════════════════════════
# snapshot original config for repeatable in-process runs
# ═══════════════════════════════════════════════════════════════════════════
_ORIGINAL_SWARM_POSITIONS = config_initial_swarm_positions.copy()
_ORIGINAL_CENTRAL_POINT = config_central_point.copy()


# ═══════════════════════════════════════════════════════════════════════════
# algorithm-free scenario preparation (used by run_case and --setup-only)
# ═══════════════════════════════════════════════════════════════════════════
def prepare_scenario(N, D, terrain_name, seed_idx, damage_mode='random',
                     output_dir=None):
    """Algorithm-free scenario preparation — no Swarm, no solve, no GPU.

    Sets up terrain, init positions, destroy list, and K(0).
    Does NOT import Swarm or run any recovery algorithm.

    Returns
    -------
    scenario : dict with keys:
        terrain_name, seed_idx, D, N, terrain_seed, terrain,
        destroy_list, positions (after z-projection + repositioning),
        K0, n_remain, valid_init, horizon, omega_max,
        init_repositioned, init_boundary_count, init_obstacle_count,
        init_steep_count, terrain_dir (or None)
    """
    import random as _random_mod

    # -- guard: N must match the imported Configurations value --
    if N != config_num_of_agents:
        raise ValueError(
            f'N={N} != config_num_of_agents={config_num_of_agents}. '
            f'Pass --num-agents {N} to the CLI or set N consistently.')

    # -- restore original positions (config globals are mutated during setup) --
    config_initial_swarm_positions[:] = _ORIGINAL_SWARM_POSITIONS
    config_central_point[:] = _ORIGINAL_CENTRAL_POINT

    # -- seeds --
    actual_seed = config_random_seed[seed_idx]
    terrain_seed = actual_seed + 9999
    # Use a dedicated Random instance to avoid interference with any
    # global random state.  run_case calls random.seed(actual_seed+1)
    # but nothing between that and stochastic_destroy consumes random,
    # so Random(actual_seed+1).sample(...) produces the same destroy list.
    _scenario_rng = _random_mod.Random(actual_seed + 1)

    # -- terrain --
    bounds = np.array([[0.0, config_width], [0.0, config_length]])
    terrain = TerrainMap(terrain_name, bounds, seed=terrain_seed)

    # -- z-projection --
    for i in range(N):
        config_initial_swarm_positions[i, 2] = \
            terrain.height(config_initial_swarm_positions[i, :2])
    config_central_point[2] = terrain.height(config_central_point[:2])

    # -- init-state repositioning (exact copy of run_case L362-424) --
    _search_radii = np.arange(1, 61) * 2.5
    init_repositioned = []
    _slope_limit = UGV_MAX_SLOPE_DEG
    init_boundary_count = 0
    init_obstacle_count = 0
    init_steep_count = 0
    _infeasible = set()
    for i in range(N):
        xy = config_initial_swarm_positions[i, :2]
        if (not _inside_shrunk_bounds(xy, terrain, UGV_AGENT_RADIUS)
                or footprint_hits_obstacle(xy, terrain, UGV_AGENT_RADIUS)
                or _footprint_max_slope(xy, terrain, UGV_AGENT_RADIUS) > _slope_limit):
            _infeasible.add(i)
            if not _inside_shrunk_bounds(xy, terrain, UGV_AGENT_RADIUS):
                init_boundary_count += 1
            elif footprint_hits_obstacle(xy, terrain, UGV_AGENT_RADIUS):
                init_obstacle_count += 1
            else:
                init_steep_count += 1

    _repositioned_set = set()
    for i in sorted(_infeasible):
        xy = config_initial_swarm_positions[i, :2].copy()
        orig_xy = xy.copy()
        found = False
        for r in _search_radii:
            for angle in np.linspace(0, 2 * np.pi, 12):
                cand = xy + np.array([r * np.cos(angle),
                                      r * np.sin(angle)])
                if not _inside_shrunk_bounds(cand, terrain, UGV_AGENT_RADIUS):
                    continue
                if footprint_hits_obstacle(cand, terrain, UGV_AGENT_RADIUS):
                    continue
                if _footprint_max_slope(cand, terrain, UGV_AGENT_RADIUS) > _slope_limit:
                    continue
                ok = True
                for j in range(N):
                    if j == i:
                        continue
                    if j in _infeasible and j not in _repositioned_set:
                        continue
                    d = np.linalg.norm(
                        cand - config_initial_swarm_positions[j, :2])
                    if d < 2.0 * UGV_AGENT_RADIUS:
                        ok = False
                        break
                if not ok:
                    continue
                config_initial_swarm_positions[i, 0] = cand[0]
                config_initial_swarm_positions[i, 1] = cand[1]
                config_initial_swarm_positions[i, 2] = terrain.height(cand)
                init_repositioned.append(
                    dict(agent=i, orig_xy=orig_xy.tolist(),
                         new_xy=cand.tolist()))
                _repositioned_set.add(i)
                found = True
                break
            if found:
                break
        if not found:
            raise RuntimeError(
                f'agent {i} could not be repositioned after '
                f'{len(_search_radii)} search radii')

    # -- terrain save --
    terrain_subdir = (f'{terrain_name}_N{N}_s{terrain_seed}')
    if output_dir:
        terrain_dir = os.path.join(output_dir, 'terrain', terrain_subdir)
        terrain.save(terrain_dir)
    else:
        terrain_dir = None

    # -- damage --
    max_d = N - config_minimum_remain_num
    if D < 1 or D > max_d:
        raise ValueError(f'D={D} out of range [1, {max_d}] (N={N})')
    if damage_mode == 'random':
        destroy_list = sorted(_scenario_rng.sample(range(N), D))
    else:
        # area damage not supported in algorithm-free mode;
        # run_case handles it after creating Environment
        destroy_list = None  # caller must handle

    # -- K(0) --
    positions_snapshot = config_initial_swarm_positions.copy()
    if destroy_list is not None:
        _destroyed = set(destroy_list)
        remain = [i for i in range(N) if i not in _destroyed]
    else:
        remain = list(range(N))
    nc, _, _ = compute_graph_state(
        positions_snapshot, remain,
        communication_range=config_communication_range)

    # -- horizon --
    horizon = int(0.8 * config_width)

    # -- omega_max --
    omega_max = UGV_OMEGA_MAX

    return dict(
        terrain_name=terrain_name,
        seed_idx=seed_idx,
        D=D,
        N=N,
        terrain_seed=terrain_seed,
        terrain=terrain,
        destroy_list=destroy_list,
        positions=positions_snapshot,
        remain=remain,
        K0=nc,
        n_remain=len(remain),
        valid_init=(destroy_list is not None and nc > 1),
        horizon=horizon,
        omega_max=omega_max,
        init_repositioned=init_repositioned,
        init_boundary_count=init_boundary_count,
        init_obstacle_count=init_obstacle_count,
        init_steep_count=init_steep_count,
        terrain_dir=terrain_dir,
    )


# ═══════════════════════════════════════════════════════════════════════════
# single-case runner  (reusable from benchmark scripts)
# ═══════════════════════════════════════════════════════════════════════════
def run_case(algo_name, N, D, terrain_name, seed_idx,
             horizon=None, omega_max=None, damage_mode='random',
             output_dir=None, save_output=True, verbose=True,
             destroy_list_override=None):
    """Run one UGV recovery case and return (metadata, trajectory, summary, saved_paths).

    Parameters are explicit so benchmark scripts can call this directly
    without relying on CLI globals or Configurations.py side effects.

    If destroy_list_override is provided, damage generation is skipped and
    the given list is used instead (for matched-destruction benchmark runs).

    Returns
    -------
    metadata : dict
    trajectory : list of dict or None
    summary : dict
    saved_paths : list of str  (empty if save_output=False)
    """
    # -- algorithm-free scenario preparation --
    sc = prepare_scenario(N, D, terrain_name, seed_idx, damage_mode,
                          output_dir)

    # -- algorithm-specific seeds (after prepare_scenario to keep random
    #    state clean for env.stochastic_destroy) --
    actual_seed = config_random_seed[seed_idx]
    terrain_seed = sc['terrain_seed']
    np.random.seed(actual_seed)
    random.seed(actual_seed + 1)
    import torch
    torch.manual_seed(actual_seed)

    # -- resolve algorithm --
    if algo_name not in ALGO_MODE:
        raise ValueError(f'Unknown algorithm "{algo_name}". '
                         f'Known: {list(ALGO_MODE)}')
    algo_mode = ALGO_MODE[algo_name]

    # -- enforce invariants (§2.1) --
    if algo_name == 'MORPHICA' and config_morphica_ablation != 'full':
        raise ValueError('UGV main results require MORPHICA ablation=full')

    # -- validation --
    if config_dimension != 3:
        raise ValueError('UGV experiment requires config_dimension=3')
    if config_communication_range <= 0:
        raise ValueError('communication_range must be > 0')

    # -- scenario-derived settings (allow explicit overrides) --
    if horizon is None:
        horizon = sc['horizon']
    if omega_max is None:
        omega_max = sc['omega_max']

    terrain = sc['terrain']
    terrain_dir = sc['terrain_dir']
    init_repositioned = sc['init_repositioned']
    init_on_boundary_count = sc['init_boundary_count']
    init_on_obstacle_count = sc['init_obstacle_count']
    init_on_steep_count = sc['init_steep_count']

    if init_repositioned and verbose:
        print(f'  Repositioned {init_on_boundary_count} boundary + '
              f'{init_on_obstacle_count} obstacle + '
              f'{init_on_steep_count} steep-slope starts '
              f'({len(init_repositioned)} agents)')

    # -- env + swarm --
    env = Environment()
    env.reset()
    swarm = Swarm(algorithm_mode=algo_mode)
    swarm.reset()

    # -- damage --
    if destroy_list_override is not None:
        destroy_list = list(destroy_list_override)
        destroy_num = len(destroy_list)
    elif damage_mode == 'random':
        destroy_num, destroy_list = env.stochastic_destroy(
            mode=2, num_of_destroyed=D)
        # Sanity check: prepare_scenario used the same seed sequence
        _dl_env = sorted(int(x) for x in destroy_list)
        _dl_sc = sc['destroy_list']
        if _dl_env != _dl_sc:
            raise RuntimeError(
                f'Destroy list divergence: '
                f'env.stochastic_destroy={_dl_env}, '
                f'prepare_scenario={_dl_sc}. '
                f'random state may have been mutated between calls.')
    elif damage_mode == 'area':
        # Sweep radii and centre offsets, pick the candidate whose
        # destroyed count is closest to the requested D.
        env.make_remain_positions()
        radii = np.linspace(config_width * 0.08, config_width * 0.55, 16)
        offsets = [
            np.array([dx * config_width * 0.2, dy * config_length * 0.2])
            for dx in (-1, 1, 0) for dy in (-1, 1, 0) if not (dx == 0 and dy == 0)
        ]
        best_list = None
        best_diff = np.inf
        for centre_xy in [config_central_point[:2] + off for off in offsets]:
            for r_try in radii:
                cand_list = []
                for idx, agent_i in enumerate(env.remain_list):
                    if np.linalg.norm(
                            env.remain_positions[idx, :2] - centre_xy) <= r_try:
                        cand_list.append(agent_i)
                n_cand = len(cand_list)
                if n_cand == 0:
                    continue
                if env.num_of_remain_agents - n_cand < config_minimum_remain_num:
                    continue
                remain_after = [a for a in env.remain_list
                                if a not in cand_list]
                nc_hyp, _, _ = compute_graph_state(
                    env.environment_positions, remain_after)
                if nc_hyp <= 1:
                    continue  # doesn't fragment
                diff = abs(n_cand - D)
                if diff < best_diff:
                    best_diff = diff
                    best_list = cand_list
        if best_list is None:
            raise RuntimeError(
                f'area damage could not produce a fragmented case '
                f'with ≈{D} destroyed agents')
        destroy_list = env.stochastic_destroy(
            mode=4, real_destroy_list=best_list)[1]
        destroy_num = len(destroy_list)

    swarm.destroy_happens(destroy_list, env.environment_positions)
    swarm.make_remain_positions()

    _destroyed_set = set(int(x) for x in destroy_list)
    _surviving_repositioned = [r for r in init_repositioned
                               if r['agent'] not in _destroyed_set]

    if verbose:
        print(f'Case: {algo_name}  N={N}  D={destroy_num}  '
              f'terrain={terrain_name}  seed_index={seed_idx}')
        print(f'  remain={len(swarm.remain_list)}  horizon={horizon}  '
              f'omega_max={omega_max:.3f}  dt={UGV_DT}  v_max={UGV_V_MAX}')
        print(f'  connectivity_model=cr_only_euclidean_3d  '
              f'one_shot={algo_mode in ONE_SHOT_MODES}')

    # -- run -- (always record trajectory for metrics; save_output gates disk)
    t0 = time.time()
    step, _, nc, traj, summary = solve_ugv(
        swarm, terrain, horizon, algo_mode,
        omega_max=omega_max, record_trajectory=True, verbose=verbose)
    elapsed = time.time() - t0

    if summary.get('invalid_initial_graph'):
        status = 'INVALID (K(0)=1)'
    elif summary.get('invalid_initial_positions'):
        status = 'INVALID (init positions)'
    elif summary['success']:
        status = 'CONNECTED'
    else:
        status = 'FAILED'

    if verbose:
        print(f'  Result: {status}  T_conn={summary["T_conn"]}  '
              f'K(0)={summary.get("num_components_initial", "?")}  '
              f'elapsed={elapsed:.1f}s')
        print(f'  Diagnostics: blocked={summary["blocked_steps"]}  '
              f'collision={summary["collision_steps"]}  '
              f'sat={summary["control_saturation_ratio"]:.3f}  '
              f'path_len={summary["path_length_mean"]:.1f}m/agent  '
              f'cost/m={summary["mean_cost_per_meter"]:.2f}  '
              f'excess_cost={summary["excess_terrain_cost_total"]:.0f}')

    # -- metadata --
    meta = dict(
        algorithm=algo_name, mode=algo_mode, terrain=terrain_name,
        N=int(N), D=int(destroy_num), requested_D=int(D),
        seed_index=seed_idx, seed_value=int(actual_seed),
        terrain_seed=int(terrain_seed),
        terrain_resolution=float(terrain.resolution),
        dt=UGV_DT, v_max=UGV_V_MAX, omega_max=omega_max,
        agent_radius=UGV_AGENT_RADIUS, max_slope_deg=UGV_MAX_SLOPE_DEG,
        communication_range=int(config_communication_range),
        communication_radius_semantics='inherited_nominal_topology',
        connectivity_model='cr_only_euclidean_3d',
        safety_controller=dict(
            safety_margin=0.3,
            n_yaw=7,
            speed_fracs=[1.0, 0.5],
            scoring_weights=dict(w_dir=0.55, w_prog=0.20, w_cost=0.15,
                                 w_slope=0.10),
            clearance='hard_only_via_dilated_mask',
            cost_max_nominal=5.0,
            collision_handling='diagnostic_only',
        ),
        damage_mode=damage_mode,
        init_repositioned=init_repositioned,
        init_repositioned_surviving=len(_surviving_repositioned),
        init_boundary_count=init_on_boundary_count,
        init_steep_count=init_on_steep_count,
        init_obstacle_count=init_on_obstacle_count,
        destroy_list=sorted(int(x) for x in destroy_list),
        destroy_count=int(destroy_num),
        benchmark_config=dict(
            constant_speed=float(config_constant_speed),
            morphica_ablation=config_morphica_ablation,
            central_point_z=float(config_central_point[2]),
        ),
        horizon=horizon, summary=summary,
        elapsed_s=round(elapsed, 3),
    )
    if terrain_dir:
        meta['terrain_dir'] = os.path.relpath(terrain_dir, output_dir)

    # -- save --
    saved_paths = []
    if save_output and output_dir and traj is not None:
        pre = (f'case_{terrain_name}_{damage_mode}_'
               f'{algo_name}_N{N}_D{destroy_num}_'
               f'r{D}_'
               f'cr{int(config_communication_range)}_'
               f'w{omega_max:.1f}_seed{seed_idx}')
        if horizon != int(0.8 * config_width):
            pre += f'_h{horizon}'
        saved_paths = list(save_trajectory(traj, meta, output_dir, pre,
                                           verbose=verbose))

    return meta, traj, summary, saved_paths


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def main():
    _p = argparse.ArgumentParser(add_help=False)
    _p.add_argument('--terrain', type=str, default='flat',
                    choices=['flat', 'rolling', 'ridge', 'roughland',
                             'mixed', 'maze'])
    _p.add_argument('--case-index', type=int, default=0)
    _p.add_argument('--output-dir', type=str, default='data/logs/ugv/cases')
    _p.add_argument('--omega-max', type=float, default=None)
    _p.add_argument('--horizon', type=int, default=None)
    _p.add_argument('--damage-mode', type=str, default='random',
                    choices=['random', 'area'])
    _ugv_args, _ = _p.parse_known_args()

    algo_name = config_algorithm_name
    if algo_name is None:
        print('ERROR: --algorithm is required (e.g. --algorithm MORPHICA)')
        return 1

    try:
        meta, traj, summary, _saved = run_case(
            algo_name=algo_name,
            N=config_num_of_agents,
            D=config_num_destructed_UAVs,
            terrain_name=_ugv_args.terrain,
            seed_idx=_ugv_args.case_index,
            horizon=_ugv_args.horizon,
            omega_max=_ugv_args.omega_max,
            damage_mode=_ugv_args.damage_mode,
            output_dir=_ugv_args.output_dir,
            save_output=True,
            verbose=True,
        )
    except ValueError as e:
        print(f'ERROR: {e}')
        return 1
    except RuntimeError as e:
        print(f'ERROR: {e}')
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())

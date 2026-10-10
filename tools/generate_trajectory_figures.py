#!/usr/bin/env python3
"""Generate representative trajectory figures for MORPHICA v2 paper.

Usage:
    python tools/generate_trajectory_figures.py              # all figures
    python tools/generate_trajectory_figures.py --fig A      # single figure
    python tools/generate_trajectory_figures.py --list       # list available figures

Output: docs/figures/fig_{A,B,C,D,E,F}.png

Each figure is reproducible: seed, N, destroyed, algorithm mode are fixed.
Per-step statistics (subnet count, force magnitudes, gap_gate, etc.) are
logged to docs/figures/trajectory_data/ for later re-plotting.
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Configurations import *
import Utils
from Environment import Environment
from Swarm import Swarm

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "figures")
DATA_DIR = os.path.join(FIG_DIR, "trajectory_data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Use Agg backend for headless rendering
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})


def setup_config(N):
    """Set Configurations for given N."""
    config_num_of_agents = N
    import pandas as pd
    pos = pd.read_excel(f"Configurations/swarm_positions_{N}.xlsx").values[:, 1:4]
    config_initial_swarm_positions = np.array(pos, dtype=np.float64)
    range_map = {20: 320, 50: 500, 100: 750, 200: 1000, 500: 1600, 1000: 2250}
    w = range_map[N]
    config_width = w; config_length = w
    config_space_range = np.array([w, w, 100.0])
    config_central_point = config_space_range * 0.5
    # Inject into module
    import Configurations as C
    C.config_num_of_agents = N
    C.config_initial_swarm_positions = config_initial_swarm_positions
    C.config_width = w; C.config_length = w
    C.config_space_range = config_space_range
    C.config_central_point = config_central_point
    return N


def run_case_with_logging(N, destroyed, seed_idx, max_step=None):
    """Run a single case using Swarm.solve()-style loop with per-step logging."""
    setup_config(N)
    if max_step is None:
        max_step = int(0.8 * config_width)

    config_num_destructed_UAVs = destroyed

    env = Environment()
    swarm = Swarm(algorithm_mode=15, use_pretrained=False)
    env.reset()
    swarm.reset()

    seed_val = config_random_seed[seed_idx] + 1718
    np.random.seed(seed_val)
    import random
    random.seed(seed_val)

    destroy_num, destroy_list = env.stochastic_destroy(
        mode=2, num_of_destroyed=destroyed
    )
    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(env.environment_positions))

    trajectory = {
        'N': N, 'destroyed': destroyed, 'seed_idx': seed_idx,
        'seed_val': seed_val, 'steps': [],
    }

    for step in range(max_step):
        # Log pre-step state
        algo = swarm.morphica_v2
        algo.check_subnets(swarm.true_positions, swarm.remain_list)

        # Compute force magnitudes
        F_anchor, w_anchor = algo.F_anchor(swarm.true_positions)
        F_scout, w_scout = algo.F_scout(swarm.true_positions, swarm.remain_list)

        f_anchor_mags = [np.linalg.norm(F_anchor[i]) for i in swarm.remain_list]
        f_scout_mags = [np.linalg.norm(F_scout[i]) for i in swarm.remain_list]
        gap_active = sum(1 for i in swarm.remain_list if np.linalg.norm(F_scout[i]) > 0.01)

        step_data = {
            'step': step,
            'num_subnets': algo.num_of_subnets,
            'subnet_sizes': [len(idxs) for idxs in algo.index_of_subnets],
            'mean_F_anchor': float(np.mean(f_anchor_mags)) if f_anchor_mags else 0,
            'max_F_anchor': float(np.max(f_anchor_mags)) if f_anchor_mags else 0,
            'mean_F_scout': float(np.mean(f_scout_mags)) if f_scout_mags else 0,
            'max_F_scout': float(np.max(f_scout_mags)) if f_scout_mags else 0,
            'gap_gate_active_ratio': float(gap_active / max(1, len(swarm.remain_list))),
            'remain_count': len(swarm.remain_list),
        }
        trajectory['steps'].append(step_data)

        # Take step (same as Swarm.solve)
        actions, _ = swarm.take_actions()
        positions = swarm.true_positions + actions * config_constant_speed
        swarm.update_true_positions(positions)
        swarm.make_remain_positions()

        connected_flag, num = swarm.check_number_of_clusters()
        if num == 1:
            trajectory['connect_step'] = step + 1
            break
    else:
        trajectory['connect_step'] = max_step

    return trajectory


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def fig_A_success_case():
    """A+S recovery: subnet count over time for a typical successful case."""
    N, destroyed, seed = 200, 50, 2  # fast-converging case
    traj = run_case_with_logging(N, destroyed, seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    steps = [s['step'] for s in traj['steps']]
    subnets = [s['num_subnets'] for s in traj['steps']]
    f_anchor = [s['mean_F_anchor'] for s in traj['steps']]
    f_scout = [s['mean_F_scout'] for s in traj['steps']]

    ax1.plot(steps, subnets, 'b-', linewidth=1.5)
    ax1.set_xlabel('Step'); ax1.set_ylabel('Subnet count')
    ax1.set_title(f'MORPHICA-A+S recovery (N={N}, d={destroyed}, seed={seed})')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, f_anchor, 'r-', label='|F_anchor|', linewidth=1)
    ax2.plot(steps, f_scout, 'b-', label='|F_scout|', linewidth=1)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Force magnitude')
    ax2.set_title('Force channels over time')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Fig A: Successful recovery (connect_step={traj["connect_step"]})',
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_A_success_case.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")
    return traj


def fig_B_type1_fragments():
    """Type I: Multi-fragment sequential merge — top slow seed."""
    N, destroyed, seed = 200, 50, 108  # BM top slow for d50
    traj = run_case_with_logging(N, destroyed, seed)

    fig, ax = plt.subplots(figsize=(10, 5))
    steps = [s['step'] for s in traj['steps']]
    subnets = [s['num_subnets'] for s in traj['steps']]

    # Find subnet count change points
    changes = []
    prev = subnets[0]
    for i, s in enumerate(subnets[1:], 1):
        if s != prev:
            changes.append((i, prev, s))
            prev = s

    ax.plot(steps, subnets, 'b-', linewidth=1.5)
    for step, old, new in changes:
        ax.axvline(x=step, color='red', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.annotate(f'{old}→{new}', (step, new), fontsize=7, color='red',
                    ha='center', va='bottom')

    ax.set_xlabel('Step'); ax.set_ylabel('Subnet count')
    ax.set_title(f'Type I: Multi-fragment merge (N={N}, d={destroyed}, seed={seed})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_B_type1_fragments.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")
    return traj


def fig_C_type2_oscillation():
    """Type II: Contact oscillation — N200 d100 top slow seed."""
    N, destroyed, seed = 200, 100, 23  # BM top slow
    traj = run_case_with_logging(N, destroyed, seed)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    steps = [s['step'] for s in traj['steps']]
    subnets = [s['num_subnets'] for s in traj['steps']]
    f_anchor = [s['mean_F_anchor'] for s in traj['steps']]
    f_scout = [s['mean_F_scout'] for s in traj['steps']]

    # Count subnet rebounds (increase = re-split)
    rebounds = []
    prev = subnets[0]
    for i, s in enumerate(subnets[1:], 1):
        if s > prev:
            rebounds.append(i)
        prev = s

    ax1.plot(steps, subnets, 'b-', linewidth=1.5)
    for r in rebounds:
        ax1.axvline(x=r, color='red', alpha=0.4, linestyle=':', linewidth=0.8)
    ax1.set_ylabel('Subnet count')
    ax1.set_title(f'Type II: Contact oscillation (N={N}, d={destroyed}, seed={seed}) — {len(rebounds)} rebounds')
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, f_anchor, 'r-', label='|F_anchor|', linewidth=0.8)
    ax2.plot(steps, f_scout, 'b-', label='|F_scout|', linewidth=0.8)
    ax2.set_xlabel('Step'); ax2.set_ylabel('Force magnitude')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_C_type2_oscillation.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")
    return traj


def fig_D_type3_slow_approach():
    """Type III: Slow approach — N200 d150 top slow seed."""
    N, destroyed, seed = 200, 150, 62
    traj = run_case_with_logging(N, destroyed, seed)

    fig, ax = plt.subplots(figsize=(10, 5))
    steps = [s['step'] for s in traj['steps']]
    subnets = [s['num_subnets'] for s in traj['steps']]
    sizes = [s['subnet_sizes'] for s in traj['steps']]

    ax.plot(steps, subnets, 'b-', linewidth=1.5)
    # Annotate subnet sizes at key points
    for t in [0, len(steps)//4, len(steps)//2, 3*len(steps)//4, len(steps)-1]:
        if t < len(sizes):
            ax.annotate(f'sizes: {sizes[t]}', (steps[t], subnets[t]),
                       fontsize=7, ha='left')

    ax.set_xlabel('Step'); ax.set_ylabel('Subnet count')
    ax.set_title(f'Type III: Slow approach (N={N}, d={destroyed}, seed={seed}, connect={traj["connect_step"]})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_D_type3_slow_approach.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")
    return traj


def fig_E_ring_center():
    """Ring-center: CR-only fundamental limit illustration."""
    N = 50
    setup_config(N)

    # Construct ring + center manually
    positions = np.zeros((N, 3))
    radius = 150.0  # > CR (120), ring and center cannot see each other
    center = np.array([config_width/2, config_length/2, 50.0])
    for i in range(N - 1):
        angle = 2 * np.pi * i / (N - 1)
        positions[i] = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
    positions[N - 1] = center  # center agent

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Initial configuration
    ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=20, alpha=0.7)
    ax1.scatter([center[0]], [center[1]], c='red', s=80, marker='*', label='anchor')
    circle = plt.Circle((center[0], center[1]), config_communication_range,
                        fill=False, color='gray', linestyle='--', alpha=0.5, label=f'CR={config_communication_range}')
    ax1.add_patch(circle)
    ax1.set_xlim(center[0] - 250, center[0] + 250)
    ax1.set_ylim(center[1] - 250, center[1] + 250)
    ax1.set_aspect('equal')
    ax1.set_title(f'Ring-center initial (r={radius} > CR={config_communication_range})')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Angular distribution at a ring cell: all neighbors at same distance
    ring_agent = positions[0]
    neighbors = positions[1:N-1]
    angles = []
    for n in neighbors:
        diff = n - ring_agent
        d = np.linalg.norm(diff)
        if d <= config_communication_range:
            angles.append(np.arctan2(diff[1], diff[0]))
    if angles:
        angles = np.array(angles)
        ax2_angles = np.linspace(-np.pi, np.pi, 100)
        ax2.scatter(angles, np.ones_like(angles), c='blue', s=30, alpha=0.7)
        # Show gaps
        sorted_a = np.sort(angles)
        gaps = np.diff(sorted_a)
        wrap_gap = 2*np.pi - (sorted_a[-1] - sorted_a[0])
        all_gaps = np.append(gaps, wrap_gap)
        largest = np.max(all_gaps)
        ax2.set_title(f'Ring cell angular distribution\nlargest_gap={largest/np.pi:.2f}π (gap_gate≈{max(0,(largest-np.pi)/np.pi):.2f})')
        ax2.axhline(y=1, color='gray', alpha=0.3)
    ax2.set_xlabel('Angle (rad)'); ax2.set_ylabel('Neighbor presence')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Fig E: Ring-Center — CR-only Fundamental Limit', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_E_ring_center.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")
    return {'N': N, 'radius': radius, 'CR': config_communication_range}


def fig_F_force_channels():
    """Force channels over time for a representative case."""
    N, destroyed, seed = 200, 100, 0  # representative d100 case
    traj = run_case_with_logging(N, destroyed, seed)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    steps = [s['step'] for s in traj['steps']]
    subnets = [s['num_subnets'] for s in traj['steps']]
    f_anchor = [s['mean_F_anchor'] for s in traj['steps']]
    f_scout = [s['mean_F_scout'] for s in traj['steps']]
    gap_ratio = [s['gap_gate_active_ratio'] for s in traj['steps']]

    axes[0].plot(steps, subnets, 'k-', linewidth=1.5)
    axes[0].set_ylabel('Subnets'); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Recovery trajectory (N={N}, d={destroyed}, seed={seed})')

    axes[1].plot(steps, f_anchor, 'r-', label='|F_anchor| (mean)', linewidth=1)
    axes[1].plot(steps, f_scout, 'b-', label='|F_scout| (mean)', linewidth=1)
    axes[1].set_ylabel('Force magnitude'); axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, gap_ratio, 'g-', linewidth=1)
    axes[2].set_ylabel('gap_gate active ratio'); axes[2].set_xlabel('Step')
    axes[2].set_ylim(0, 1.05); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Fig F: Force Channels Over Time', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig_F_force_channels.png')
    fig.savefig(path); plt.close(fig)
    print(f"  Saved {path}")
    return traj


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FIGURES = {
    'A': ('fig_A_success_case', 'Successful recovery trajectory + force channels', fig_A_success_case),
    'B': ('fig_B_type1_fragments', 'Type I: multi-fragment sequential merge', fig_B_type1_fragments),
    'C': ('fig_C_type2_oscillation', 'Type II: contact oscillation with rebounds', fig_C_type2_oscillation),
    'D': ('fig_D_type3_slow_approach', 'Type III: slow approach (large d)', fig_D_type3_slow_approach),
    'E': ('fig_E_ring_center', 'Ring-center CR-only limit (synthetic)', fig_E_ring_center),
    'F': ('fig_F_force_channels', 'Force channels over time', fig_F_force_channels),
}


def main():
    parser = argparse.ArgumentParser(description="MORPHICA v2 trajectory figure generator")
    parser.add_argument('--fig', type=str, choices=list(FIGURES.keys()), help='Generate single figure')
    parser.add_argument('--list', action='store_true', help='List available figures')
    args = parser.parse_args()

    if args.list:
        for key, (name, desc, _) in FIGURES.items():
            print(f"  Fig {key}: {name} — {desc}")
        return

    to_generate = [args.fig] if args.fig else list(FIGURES.keys())

    print(f"Generating {len(to_generate)} figure(s) → {FIG_DIR}/")
    for key in to_generate:
        name, desc, func = FIGURES[key]
        print(f"  Fig {key}: {desc}...")
        try:
            data = func()
            # Save trajectory data
            data_path = os.path.join(DATA_DIR, f'{name}.json')
            with open(data_path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
            print(f"    Data saved to {data_path}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()

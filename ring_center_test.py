"""Synthetic connected-ring test: verify bearer (P) necessity for symmetry breaking.

Constructs a single *connected* ring subnet surrounding a small central cluster.
Ring agents are dense enough that:
  - Internal gap_gate = 0 (neighbors all around → largest_gap < π)
  - Scout (spreading) provides no directional cue
  - Retraction pulls ring uniformly inward but creates no lateral asymmetry

Polarization (bearer): the ring agent closest to center gets an extra pull.
This creates a directional protrusion — a "nose" — that breaks the circular
symmetry and enables the ring to reach the center cluster.
Without bearer, the ring shrinks uniformly and stalls without ever touching.

Prediction:
  - full RPS:   converges (bearer creates directional protrusion)
  - no-P:        stalls at or near max_step (uniform contraction → deadlock)

Usage:
  python ring_center_test.py [--num-ring 40] [--center-size 5] [--max-step 800]
"""

import sys
import argparse
import numpy as np

CR = 120.0
DIM = 3
CONFIG_SPACE_RANGE = np.array([2250.0, 2250.0, 100.0])
CONFIG_CENTRAL_POINT = CONFIG_SPACE_RANGE * 0.5
CONFIG_CONSTANT_SPEED = 1.0


def build_connected_ring(num_ring=40, center_size=5):
    """A connected ring of agents surrounding a small center cluster.

    Ring: agents on a circle, connected (chord < CR) but far enough from center
    that the ring subnet is initially disconnected from the center subnet.

    Returns (positions, remain_list, center).
    """
    center = CONFIG_CENTRAL_POINT.copy()
    positions = []

    # Center cluster: small tight blob
    np.random.seed(1)
    for _ in range(center_size):
        offset = np.random.randn(DIM) * 5.0
        offset[2] *= 0.1
        positions.append(center + offset)

    # Connected ring at radius just beyond CR from center.
    # Ring agents densely packed so chord(neighbor, neighbor) << CR.
    ring_radius = CR * 3.5  # ~420 — retraction takes ~300 steps just to reach CR distance

    for i in range(num_ring):
        angle = 2 * np.pi * i / num_ring
        x = center[0] + ring_radius * np.cos(angle)
        y = center[1] + ring_radius * np.sin(angle)
        z = center[2]
        positions.append(np.array([x, y, z]))

    positions = np.array(positions, dtype=np.float64)
    remain_list = list(range(len(positions)))
    return positions, remain_list, center


def check_connectivity(positions, remain_list):
    """BFS: returns (is_connected, num_components)."""
    remain_pos = np.array([positions[i] for i in remain_list])
    n = len(remain_pos)
    G = np.matmul(remain_pos, remain_pos.T)
    H = np.tile(np.diag(G), (n, 1))
    D = np.sqrt(H + H.T - 2 * G)
    A = D <= CR
    np.fill_diagonal(A, False)
    visited = np.zeros(n, dtype=bool)
    n_comp = 0
    for start in range(n):
        if visited[start]:
            continue
        n_comp += 1
        stack = [start]
        visited[start] = True
        while stack:
            v = stack.pop()
            nb = np.where(A[v] & ~visited)[0]
            visited[nb] = True
            stack.extend(nb.tolist())
    return n_comp == 1, n_comp


def run_test(ablation, positions, remain_list, max_step=800):
    """Run MORPHICA variant, return (step, final_ncomp, trajectory, final_positions)."""
    from DIRA_Algorithm.MORPHICA import MORPHICA

    morphica = MORPHICA(positions, ablation=ablation)
    current = positions.copy()
    last_ncomp = None
    trajectory = []  # list of (step, n_comp)

    for step in range(max_step):
        speed, info = morphica.group_fusion_step(current, remain_list)
        current = current + speed * CONFIG_CONSTANT_SPEED
        connected, n_comp = check_connectivity(current, remain_list)

        if n_comp != last_ncomp:
            print(f"  [{ablation:15s}] step {step:4d}: subnets={n_comp:2d}")
            last_ncomp = n_comp
        trajectory.append((step, n_comp))

        if connected:
            print(f"  [{ablation:15s}] step {step:4d}: CONVERGED")
            return step, 1, trajectory, current.copy()

    print(f"  [{ablation:15s}] step {max_step:4d}: STALLED ({last_ncomp} subnets)")
    return max_step, last_ncomp, trajectory, current.copy()


def main():
    parser = argparse.ArgumentParser(description="Connected-ring symmetry test")
    parser.add_argument('--num-ring', type=int, default=40)
    parser.add_argument('--center-size', type=int, default=5)
    parser.add_argument('--max-step', type=int, default=2000)
    args = parser.parse_args()

    import os, json
    log_dir = "Logs/ring_center_test"
    os.makedirs(log_dir, exist_ok=True)

    positions, remain_list, center = build_connected_ring(
        num_ring=args.num_ring, center_size=args.center_size
    )

    connected, n_comp = check_connectivity(positions, remain_list)
    print(f"Initial: {len(remain_list)} agents, {n_comp} subnets "
          f"(expected: 2 = ring + center), connected={connected}")
    assert not connected, "Initial state must be disconnected"
    assert n_comp == 2, f"Expected 2 subnets, got {n_comp}"

    chord = 2 * (CR * 3.5) * np.sin(np.pi / args.num_ring)
    print(f"  Ring radius={CR*3.5:.0f}, chord≈{chord:.1f}, CR={CR}")

    print("\n--- Full RPS ---")
    steps_full, final_ncomp_full, traj_full, pos_full = run_test('full', positions, remain_list, args.max_step)

    print("\n--- no-P (no bearer) ---")
    steps_noP, final_ncomp_noP, traj_noP, pos_noP = run_test('no-P', positions, remain_list, args.max_step)

    passed = False
    print("\n" + "=" * 60)
    print(f"Results (max_step={args.max_step}):")
    print(f"  full RPS:  {steps_full:4d} steps  "
          f"{'✓ CONVERGED' if steps_full < args.max_step else f'✗ STALLED ({final_ncomp_full} subnets)'}")
    print(f"  no-P:      {steps_noP:4d} steps  "
          f"{'✓ CONVERGED' if steps_noP < args.max_step else f'✗ STALLED ({final_ncomp_noP} subnets)'}")
    print()

    if steps_full < args.max_step and steps_noP >= args.max_step:
        print("PASS: Bearer is necessary for symmetry breaking in connected-ring.")
        passed = True
    elif steps_full < args.max_step and steps_noP < args.max_step:
        ratio = steps_noP / max(1, steps_full)
        if ratio >= 2.0:
            print(f"PASS (weak): no-P converges but {ratio:.1f}× slower "
                  f"({steps_noP} vs {steps_full}). Bearer clearly accelerates.")
            passed = True
        else:
            print(f"INCONCLUSIVE: slowdown only {ratio:.1f}×. "
                  f"Symmetry may not be strong enough at this ring density.")
    elif steps_full >= args.max_step:
        print("FAIL: Full RPS did not converge. Ring radius may be too large, "
              "or max_step too small.")
    else:
        print("FAIL: Unexpected outcome.")

    # --- Save log ---
    config_key = f"R{args.num_ring}_C{args.center_size}"
    log_path = os.path.join(log_dir, f"ring_center_{config_key}.txt")
    with open(log_path, 'w') as f:
        f.write(f"# Ring-Center Symmetry Breaking Test\n")
        f.write(f"# num_ring={args.num_ring} center_size={args.center_size} "
                f"max_step={args.max_step} CR={CR}\n")
        f.write(f"# ring_radius={CR*3.5:.0f} chord={chord:.1f}\n")
        f.write(f"# full_RPS: steps={steps_full} "
                f"{'converged' if steps_full < args.max_step else 'stalled'}\n")
        f.write(f"# no_P: steps={steps_noP} "
                f"{'converged' if steps_noP < args.max_step else 'stalled'}\n")
        f.write(f"# passed={passed}\n")

        # Initial positions
        f.write(f"\n[initial_positions]\n")
        f.write("agent_id\tx\ty\tz\n")
        for i, p in enumerate(positions):
            f.write(f"{i}\t{p[0]:.6f}\t{p[1]:.6f}\t{p[2]:.6f}\n")

        # Final positions: full RPS
        f.write(f"\n[final_positions_full_RPS]\n")
        f.write("agent_id\tx\ty\tz\n")
        for i, p in enumerate(pos_full):
            f.write(f"{i}\t{p[0]:.6f}\t{p[1]:.6f}\t{p[2]:.6f}\n")

        # Final positions: no-P
        f.write(f"\n[final_positions_noP]\n")
        f.write("agent_id\tx\ty\tz\n")
        for i, p in enumerate(pos_noP):
            f.write(f"{i}\t{p[0]:.6f}\t{p[1]:.6f}\t{p[2]:.6f}\n")

        # Trajectory
        f.write(f"\n[trajectory]\n")
        f.write("step\tfull_ncomp\tnoP_ncomp\n")
        d_full = {s: n for s, n in traj_full}
        d_noP = {s: n for s, n in traj_noP}
        all_steps = sorted(set(d_full.keys()) | set(d_noP.keys()))
        last_f, last_n = 2, 2
        for s in all_steps:
            last_f = d_full.get(s, last_f)
            last_n = d_noP.get(s, last_n)
            f.write(f"{s}\t{last_f}\t{last_n}\n")
    print(f"\nTrajectory saved to: {log_path}")

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

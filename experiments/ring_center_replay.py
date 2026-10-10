"""Replay the ring-centre diagnostic for the full controller and the no-P ablation.

Both variants run on one explicit shared damage mask, so the comparison isolates
the effect of Polarization. Results are written to
`data/control_suites/diagnostics/`.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).resolve().parents[1] / 'data' / 'control_suites' / 'diagnostics'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, default=129,
                        help='seed index defining the damage mask (default: 129)')
    args = parser.parse_args()

    import numpy as np
    from experiments.run_control_suites import configure, labels
    from tools.damage_mask import damage_mask, damage_seed

    C = configure(1000)
    from controllers.MORPHICA import MORPHICA

    C._obs_packet_loss = C._obs_delay_steps = C._obs_position_noise = 0
    initial = C.config_initial_swarm_positions.copy()
    destroyed = damage_mask(1000, 500, args.case)
    remain = [i for i in range(1000) if i not in set(destroyed)]
    horizon = int(0.8 * C.config_width)
    seed = damage_seed(args.case)

    results = {'case': args.case, 'seed': seed, 'N': 1000, 'D': 500,
               'horizon': horizon, 'initial_positions': initial,
               'destroy_list': np.array(destroyed), 'remain': np.array(remain)}

    for variant in ['full', 'no-P']:
        positions = initial.copy()
        controller = MORPHICA(initial, ablation=variant, observation_seed=seed)
        curve = [(0, labels(positions[remain])[0])]
        for step in range(horizon):
            actions, _ = controller.group_fusion_step(positions, remain)
            positions += actions * C.config_constant_speed
            k, _ = labels(positions[remain])
            curve.append((step + 1, k))
            if k == 1:
                break
        event = k == 1 and step < horizon - 1
        results[variant + '_T_conn'] = step
        results[variant + '_success'] = event
        results[variant + '_trajectory_K'] = np.array(curve)
        results[variant + '_final_positions'] = positions
        print(f'{variant}: index={step}, K={k}, success={event}', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    artifact = OUT / f'ring_center_case{args.case}.npz'
    np.savez_compressed(artifact, **results)
    manifest = {
        'artifact': str(artifact.relative_to(Path(__file__).resolve().parents[1])),
        'case': args.case,
        'seed': seed,
        'mask_sha256': hashlib.sha256(json.dumps(sorted(destroyed)).encode()).hexdigest(),
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    artifact.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(artifact, flush=True)


if __name__ == '__main__':
    main()

"""Replay a historical full/no-P diagnostic with an explicit shared damage mask.

Writes new artifacts under Logs/manuscript_revision/diagnostics, preserving the
legacy caches. Their labels do not reliably identify the actual damage mask.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random


def damage_mask(n, d, case):
    # Controller imports and construction can reseed the global random module.
    return random.Random(case + 1725).sample(range(n), d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=int, default=129)
    args = parser.parse_args()
    import numpy as np
    from tools.run_manuscript_evidence import configure, labels
    C = configure(1000)
    from DIRA_Algorithm.MORPHICA import MORPHICA
    C._E9_packet_loss = C._E9_delay_steps = C._E9_position_noise = 0
    initial = C.config_initial_swarm_positions.copy()
    destroyed = damage_mask(1000, 500, args.case)
    remain = [i for i in range(1000) if i not in set(destroyed)]
    horizon = int(.8 * C.config_width)
    seed = args.case + 1725
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
    out = Path('Logs/manuscript_revision/diagnostics')
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / f'case{args.case}_replay.npz'
    np.savez_compressed(artifact, **results)
    manifest = {'artifact': str(artifact), 'case': args.case, 'seed': seed,
                'mask_sha256': hashlib.sha256(json.dumps(sorted(destroyed)).encode()).hexdigest(),
                'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                'sources': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                            for p in ['record_case129.py', 'DIRA_Algorithm/MORPHICA.py',
                                      'Configurations/swarm_positions_1000.xlsx']}}
    artifact.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(artifact, flush=True)


if __name__ == '__main__':
    main()

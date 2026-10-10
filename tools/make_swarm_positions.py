"""Generate an initial swarm configuration file.

Samples a random configuration that is connected under the communication range
and writes it in the column layout read by `Configurations.py`: a leading index
column followed by x, y, z.

Usage:
    python tools/make_swarm_positions.py --num-agents 1000 --side 2250
"""

import argparse
import pathlib
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import Utils

# Arena side length used for each swarm size in the reported benchmarks.
DEFAULT_SIDE = {'20': 320.0, '50': 500.0, '100': 750.0, '200': 1000.0,
                '500': 1600.0, '1000': 2250.0, '2000': 3200.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-agents', type=int, required=True)
    parser.add_argument('--side', type=float, default=None,
                        help='arena side length (default: the benchmark value for this swarm size)')
    parser.add_argument('--height', type=float, default=100.0)
    parser.add_argument('--seed', type=int, default=0, help='sampling seed')
    parser.add_argument('--out', type=pathlib.Path, default=None)
    args = parser.parse_args()

    side = args.side or DEFAULT_SIDE.get(str(args.num_agents))
    if side is None:
        parser.error('--side is required for a swarm size without a benchmark default')

    rng = random.Random(args.seed)
    while True:
        positions = np.array([[rng.uniform(1, side), rng.uniform(1, side), rng.uniform(1, args.height)]
                              for _ in range(args.num_agents)])
        connected, _ = Utils.check_if_a_connected_graph(positions, len(positions))
        if connected:
            break

    out = args.out or (pathlib.Path(__file__).resolve().parents[1] / 'data' / 'configs'
                       / f'swarm_positions_{args.num_agents}.xlsx')
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(positions, columns=['x', 'y', 'z']).to_excel(out, index=True)
    print(f'{args.num_agents} agents, side {side:g} -> {out}')


if __name__ == '__main__':
    main()

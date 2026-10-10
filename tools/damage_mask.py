"""Canonical damage masks for every benchmark case.

A case is identified by its seed index `c`. At swarm size `N` with `D` agents
removed, the removal set is deterministic:

    random.Random(c + 1725).sample(range(N), D)

`c + 1725` is the seed used by the original sweep runner
(`config_random_seed[c] + 1718` with `config_random_seed = range(7, 10007)`), so
the same index reproduces the same damage mask across every method, scale and
communication radius. The function is kept separate from any controller because
constructing a controller may reseed the global `random` module.
"""

import random


def damage_seed(case: int) -> int:
    """Seed value used for case `case`."""
    return case + 1725


def damage_mask(n: int, d: int, case: int) -> list:
    """Removal set for `case`: `d` agents removed from an `n`-agent swarm."""
    return random.Random(damage_seed(case)).sample(range(n), d)

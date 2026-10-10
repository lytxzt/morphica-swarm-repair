"""Finite synchronous flooding diagnostic for MORPHICA's aggregate limit.

Each motion step freezes the current contact graph for ``relay_rounds`` message
rounds. Identifier-tagged position records travel one edge per round; repeated
records are deduplicated. Each node reads R/P from its own reachable records.
This tests finite graph propagation, not a physical network's latency or load.
"""

import numpy as np
from scipy.sparse import csr_matrix

from DIRA_Algorithm.MORPHICA import MORPHICA, EPS
from Configurations import config_dimension, config_communication_range


class FiniteRelayMORPHICA(MORPHICA):
    def __init__(self, *args, relay_rounds, **kwargs):
        super().__init__(*args, **kwargs)
        if relay_rounds < 0:
            raise ValueError('relay_rounds must be nonnegative')
        self.relay_rounds = int(relay_rounds)

    def check_subnets(self, node_global_positions, remain_list):
        super().check_subnets(node_global_positions, remain_list)
        n = len(remain_list)
        reach = np.eye(n, dtype=bool)
        adj = csr_matrix(self.A_mat.astype(np.int32))
        for _ in range(self.relay_rounds):
            reach |= (adj @ reach.astype(np.int32)) > 0
        self.relay_reach = reach
        self.relay_remain = np.asarray(remain_list, dtype=int)
        self.relay_positions = self.sensed_positions[self.relay_remain]

    def F_retraction(self, node_global_positions):
        force = np.zeros((self.num_of_agents, config_dimension))
        weight = np.zeros_like(force)
        for li, gi in enumerate(self.relay_remain):
            records = self.relay_positions[self.relay_reach[li]]
            centre = records.mean(axis=0)
            delta = self.anchor_region - centre
            distance = np.linalg.norm(delta)
            force[gi] = min(distance, config_communication_range) * delta / (distance + EPS)
            median = np.median(np.linalg.norm(records - self.anchor_region, axis=1))
            weight[gi] = np.clip(median / (distance + EPS), 0.0, 1.0)
        return force, weight

    def F_polarization(self, node_global_positions, K=1):
        if K != 1:
            raise ValueError('Finite-relay diagnostic implements one bearer per local view')
        force = np.zeros((self.num_of_agents, config_dimension))
        weight = np.zeros_like(force)
        for li, gi in enumerate(self.relay_remain):
            visible = np.flatnonzero(self.relay_reach[li])
            distances = np.linalg.norm(self.relay_positions[visible] - self.anchor_region, axis=1)
            # Stable agent-ID tie break is a concrete front-selection channel.
            bearer = visible[np.argmin(distances)]
            if bearer != li:
                continue
            delta = self.anchor_region - self.relay_positions[li]
            distance = np.linalg.norm(delta)
            if distance < EPS:
                continue
            force[gi] = (len(visible) / self.num_of_agents) * min(
                distance, config_communication_range
            ) * delta / distance
            weight[gi] = 1.0
        return force, weight

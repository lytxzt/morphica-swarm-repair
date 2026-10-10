from copy import deepcopy

import numpy as np

import Utils
from Configurations import *


EPS = 1e-4

# Continuous baseline support for scouting alpha terms.
# Each social term s,p,q is lifted so that even agents with low raw scores
# retain a minimum scouting contribution. This is NOT a hard gate — it is a
# continuous modulation: eff = floor + (1-floor)*raw, which preserves the
# product structure alpha_i = s(|G_i|) * p(r_i) * q(n_i) from target.md.
SCOUT_ALPHA_FLOOR = 0.35


class GroupFusion:
    def __init__(
        self,
        initial_positions,
        w_A=config_dira_nmi_w_A,
        w_S=config_dira_nmi_w_S,
        w_L=config_dira_nmi_w_L,
        ablation=config_dira_nmi_ablation,
        use_velocity_alignment=config_dira_nmi_use_velocity_alignment,
        anchor_region=None,
    ):
        self.init_positions = deepcopy(initial_positions)
        self.num_of_agents = len(self.init_positions)

        self.w_A = w_A
        self.w_S = w_S
        self.w_L = w_L
        self.ablation = ablation
        self.use_velocity_alignment = use_velocity_alignment
        self.anchor_region = deepcopy(config_central_point if anchor_region is None else anchor_region)

        self.prev_positions = None
        self.velocities = None
        self.last_step_info = {}

        self.num_of_subnets = 0
        self.positions_of_subnets = []
        self.index_of_subnets = []

    def check_subnets(self, node_global_positions, remain_list):
        remain_positions = np.array([deepcopy(node_global_positions[i]) for i in remain_list], dtype=np.float64)
        if len(remain_positions) == 0:
            self.num_of_subnets = 0
            self.positions_of_subnets = []
            self.index_of_subnets = []
            return

        A = Utils.make_A_matrix(remain_positions, len(remain_positions), config_communication_range)
        _, num = Utils.count_clusters_from_A(A)

        self.num_of_subnets = num
        self.positions_of_subnets, local_subnet_indexes = Utils.split_the_positions_into_clusters_and_indexes(
            remain_positions,
            num,
            A,
        )

        self.index_of_subnets = []
        for subnet_indexes in local_subnet_indexes:
            self.index_of_subnets.append([remain_list[i] for i in subnet_indexes])

    def F_anchor(self):
        """Recovery Anchoring: subnet-level force preserves fragment geometry.

        Instead of per-agent anchor forces (which can pull agents within the
        same fragment in slightly different directions and distort fragment
        shape), each subnet receives a single force vector and weight.
        The fragment moves as a unit toward the recovery basin while scout
        handles outward probing — this is fragment-level anchoring, not a
        hard gate.
        """
        F_anchor = np.zeros((self.num_of_agents, config_dimension))
        w_anchor = np.zeros((self.num_of_agents, config_dimension))
        anchor = self.anchor_region

        if self.ablation in ['no-anchor']:
            return F_anchor, w_anchor

        for k in range(self.num_of_subnets):
            subnet_positions = self.positions_of_subnets[k]
            subnet_center = np.mean(subnet_positions, axis=0)
            force = anchor - subnet_center
            anchor_distance = np.linalg.norm(force)
            nearest_anchor_distance = np.min(np.linalg.norm(anchor - subnet_positions, axis=1))
            group_weight = np.clip(nearest_anchor_distance / (anchor_distance + EPS), 0.0, 1.0)

            for agent_i in self.index_of_subnets[k]:
                F_anchor[agent_i] = force
                w_anchor[agent_i] = group_weight

        return deepcopy(F_anchor), deepcopy(w_anchor)

    def _scout_group_size_term(self, group_size):
        """s(|G_i|): larger groups → stronger scouting. Range [0, 1]."""
        return (group_size - 1) / max(1, self.num_of_agents - 1)

    def _scout_periphery_term(self, r_i, r_max):
        """p(r_i): agents near group periphery → stronger scouting. Range [0, 1]."""
        if r_max < EPS:
            return 0.0
        return np.clip(r_i / r_max, 0.0, 1.0)

    def _scout_neighbour_term(self, n_i):
        """q(n_i): more local neighbours → stronger scouting. Range [0, 1].

        Uses a sigmoidal form n_i/(n_i + k) for continuous modulation,
        avoiding the saturation problem of n_i/(n_i-1).
        """
        if n_i <= 0:
            return 0.0
        k = 3.0  # half-saturation constant
        return n_i / (n_i + k)

    def _compute_neighbour_counts(self, positions, indexes):
        counts = {}
        for i in indexes:
            count = 0
            for j in indexes:
                if i == j:
                    continue
                if np.linalg.norm(positions[i] - positions[j]) <= config_communication_range:
                    count += 1
            counts[i] = count
        return counts

    def F_scout(self, node_global_positions, remain_list):
        """Socially Supported Scouting: alpha_i = s(|G_i|) * p(r_i) * q(n_i).

        Direction: outward unit vector ô_i from group center to agent.
        Force F_i^scout = alpha_i * ô_i — magnitude comes from social
        modulation, not raw distance, so scout does not grow with group radius.
        """
        F_scout = np.zeros((self.num_of_agents, config_dimension))
        w_scout = np.zeros((self.num_of_agents, config_dimension))

        if self.ablation in ['no-scout', 'anchor-only']:
            return F_scout, w_scout

        for k in range(self.num_of_subnets):
            group = self.positions_of_subnets[k]
            indexes = self.index_of_subnets[k]
            group_size = len(indexes)
            if group_size <= 1:
                continue

            group_center = np.mean(group, axis=0)

            # Group radius: max distance from center (for normalizing periphery term)
            r_max = max([np.linalg.norm(group[i] - group_center) for i in range(len(group))] + [EPS])

            neighbour_counts = self._compute_neighbour_counts(node_global_positions, indexes)

            for local_i, agent_i in enumerate(indexes):
                pos = node_global_positions[agent_i]
                r_i = np.linalg.norm(pos - group_center)

                # Outward unit direction from group center to agent.
                # Magnitude is controlled by alpha_i, not by raw distance,
                # so scout force does not grow with group radius.
                scout_force = (pos - group_center) / (r_i + EPS)

                # Three modulating terms (raw values in [0, 1])
                s_raw = self._scout_group_size_term(group_size)
                p_raw = self._scout_periphery_term(r_i, r_max)
                q_raw = self._scout_neighbour_term(neighbour_counts[agent_i])

                # Continuous baseline support: lift each term so that even
                # low-scoring agents retain a minimum scouting contribution.
                # eff = floor + (1 - floor) * raw  — preserves the product
                # structure from target.md without introducing a hard gate.
                s_eff = SCOUT_ALPHA_FLOOR + (1.0 - SCOUT_ALPHA_FLOOR) * s_raw
                p_eff = SCOUT_ALPHA_FLOOR + (1.0 - SCOUT_ALPHA_FLOOR) * p_raw
                q_eff = SCOUT_ALPHA_FLOOR + (1.0 - SCOUT_ALPHA_FLOOR) * q_raw

                # Apply per-term ablations (override to unity, removing modulation)
                if self.ablation == 'no-size':
                    s_eff = 1.0
                if self.ablation == 'no-periphery':
                    p_eff = 1.0
                if self.ablation == 'no-neighbour-support':
                    q_eff = 1.0

                if self.ablation == 'constant-scouting':
                    alpha_i = 0.5
                else:
                    alpha_i = np.clip(s_eff * p_eff * q_eff, 0.0, 1.0)


                F_scout[agent_i] = scout_force
                w_scout[agent_i] = alpha_i

        return deepcopy(F_scout), deepcopy(w_scout)

    def F_align_velocity(self, node_global_positions, remain_list):
        """Neighbour Interaction: distance-based attraction/repulsion + velocity alignment.

        Each neighbour j within CR contributes two forces:
          1. Distance interaction: repulsion at close range (d < d_eq),
             attraction at far range (d > d_eq).  d_eq = CR/2 creates a
             soft boundary that prevents premature velocity coupling across
             subnet boundaries while maintaining internal coherence.
          2. Velocity alignment: Vicsek-style, weighted by quadratic distance
             falloff and direction-similarity gate (cos θ).

        The combined weighted average is fully distributed: every term uses
        only neighbour positions, velocities, and the pre-agreed CR, d_eq.
        """
        F_align = np.zeros((self.num_of_agents, config_dimension))
        w_align = np.zeros((self.num_of_agents, config_dimension))
        CR = config_communication_range
        d_eq = CR * 0.5

        if self.velocities is not None:
            velocities = self.velocities
        elif self.prev_positions is not None:
            velocities = node_global_positions - self.prev_positions
        else:
            return F_align, w_align

        for i in remain_list:
            v_i = velocities[i]
            v_i_norm = np.linalg.norm(v_i)
            if v_i_norm < EPS:
                continue

            vel_sum = np.zeros(config_dimension)
            dist_sum = np.zeros(config_dimension)
            total_w = 0.0

            for j in remain_list:
                if i == j:
                    continue
                diff = node_global_positions[j] - node_global_positions[i]
                distance = np.linalg.norm(diff)
                if distance > CR:
                    continue

                unit_ij = diff / (distance + EPS)

                # -- velocity alignment weight --
                v_j = velocities[j]
                v_j_norm = np.linalg.norm(v_j)
                if v_j_norm > EPS:
                    w_d = (distance / CR) ** 2
                    cos_theta = np.dot(v_i, v_j) / (v_i_norm * v_j_norm)
                    w_vel = w_d * max(0.0, cos_theta)
                else:
                    w_vel = 0.0

                # -- distance interaction weight --
                # w_dist ∈ [-1, +1]: negative=repulsion, positive=attraction
                w_dist = np.clip((distance - d_eq) / d_eq, -1.0, 1.0)

                w = w_vel + abs(w_dist)
                if w < EPS:
                    continue

                # velocity contribution (weighted by w_vel)
                if w_vel > EPS:
                    vel_sum += (v_j - v_i) * w_vel
                # distance contribution (weighted by w_dist, signed)
                dist_sum += unit_ij * w_dist

                total_w += w

            if total_w < EPS:
                continue

            # Weighted average: each contribution type is already pre-multiplied
            # by its per-neighbour weight; dividing by total_w normalises across
            # all neighbours.
            F_align[i] = (vel_sum + dist_sum) / total_w
            w_align[i] = total_w / max(1, len([1 for j in remain_list
                if j != i and np.linalg.norm(node_global_positions[i] - node_global_positions[j]) <= CR]))

        return deepcopy(F_align), deepcopy(w_align)

    def _apply_ablation(self, F_anchor, w_anchor, F_scout, w_scout, F_align, w_align):
        if self.ablation == 'no-anchor':
            F_anchor = np.zeros_like(F_anchor)
            w_anchor = np.zeros_like(w_anchor)
        elif self.ablation == 'no-scout':
            F_scout = np.zeros_like(F_scout)
            w_scout = np.zeros_like(w_scout)
        elif self.ablation == 'no-align':
            F_align = np.zeros_like(F_align)
            w_align = np.zeros_like(w_align)
        elif self.ablation == 'anchor-only':
            F_scout = np.zeros_like(F_scout)
            w_scout = np.zeros_like(w_scout)
            F_align = np.zeros_like(F_align)
            w_align = np.zeros_like(w_align)

        return F_anchor, w_anchor, F_scout, w_scout, F_align, w_align

    def _normalize_speed(self, speed, remain_list):
        """Always normalize to unit speed (like v1), not just cap.

        When forces nearly cancel, sub-unit speeds waste iterations.
        Always moving at full speed ensures consistent step size.
        """
        for i in remain_list:
            norm = np.linalg.norm(speed[i])
            if norm > EPS:
                speed[i] = speed[i] / norm
        return speed

    def group_fusion_step(self, node_global_positions, remain_list):
        """One step of anchoring-scouting-alignment group-fusion dynamics.

        Forces are NOT unitized — anchor magnitude encodes distance urgency,
        while scout direction is unitized and scaled by alpha_i.
        All three are combined with per-agent weights and global coefficients,
        then speed-clipped to max 1.
        """
        self.check_subnets(node_global_positions, remain_list)

        F_anchor, w_anchor = self.F_anchor()
        F_scout, w_scout = self.F_scout(node_global_positions, remain_list)
        if self.use_velocity_alignment:
            F_align, w_align = self.F_align_velocity(node_global_positions, remain_list)
        else:
            F_align = np.zeros((self.num_of_agents, config_dimension))
            w_align = np.zeros((self.num_of_agents, config_dimension))

        F_anchor, w_anchor, F_scout, w_scout, F_align, w_align = self._apply_ablation(
            F_anchor,
            w_anchor,
            F_scout,
            w_scout,
            F_align,
            w_align,
        )

        # Combine forces: anchor keeps raw magnitude (encodes urgency),
        # scout is pre-unitized (direction only, scaled by alpha_i),
        # alignment uses raw velocity differences.
        speed = (
            self.w_A * w_anchor * F_anchor
            + self.w_S * w_scout * F_scout
            + self.w_L * w_align * F_align
        )
        speed = self._normalize_speed(speed, remain_list)

        step_info = {
            'num_subnets': self.num_of_subnets,
            'subnet_sizes': [len(indexes) for indexes in self.index_of_subnets],
            'scout_alphas': deepcopy(w_scout),
            'component_count': self.num_of_subnets,
            'weights': {'w_A': self.w_A, 'w_S': self.w_S, 'w_L': self.w_L},
            'ablation': self.ablation,
            'use_velocity_alignment': self.use_velocity_alignment,
        }

        self.prev_positions = deepcopy(node_global_positions)
        self.velocities = deepcopy(speed)
        self.last_step_info = deepcopy(step_info)

        return deepcopy(speed), deepcopy(step_info)

    def dira(self, node_global_positions, remain_list):
        speed, _ = self.group_fusion_step(node_global_positions, remain_list)
        return deepcopy(speed)


DIRA_NMI = GroupFusion

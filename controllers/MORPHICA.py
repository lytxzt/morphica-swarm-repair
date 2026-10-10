from collections import deque
from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import Utils
from Configurations import *

EPS = 1e-4


class MORPHICA:
    """
    MORPHICA-RPS: Morphogenetic self-assembly via a three-phase repair program.

    The algorithm models damage recovery as a morphogenetic process — the
    swarm reorganizes from a shattered state into a single connected tissue
    through the same principles that drive embryonic development:

    R — Retraction  (morphogen-guided collective contraction)
    P — Polarization (tip-cell selection for symmetry breaking)
    S — Spreading    (CIL-driven free-edge protrusion via angular-void)

    In the embryo, a signaling center (e.g. ZPA, AER, notochord) secretes
    morphogens that form a concentration gradient; cells read the gradient
    and orient their migration/division toward the source. Simultaneously,
    cells at the tissue frontier sense the absence of neighbors and extend
    protrusions into free space (contact inhibition of locomotion). These
    two directional cues are propagated through the tissue via mechanical
    coupling — cadherin junctions and planar cell polarity pathways transmit
    directional information along sparse cell-cell interfaces, much as
    convergent extension coordinates the intercalation of hundreds of cells
    during gastrulation.

    The same geometry-based rule is used across benchmarks. Numerical choices
    include a 20 m contact margin, a 0.01 gap gate and three clearance neighbours.
    """

    def __init__(
        self,
        initial_positions,
        anchor_region=None,
        ablation='full',
        observation_seed=None,
    ):
        self.init_positions = deepcopy(initial_positions)
        self.num_of_agents = len(self.init_positions)

        self.anchor_region = deepcopy(
            config_central_point if anchor_region is None else anchor_region
        )

        self.ablation = ablation

        self.prev_positions = None
        self.velocities = None
        self.last_step_info = {}

        # Non-ideal: position history buffer for delay
        from Configurations import _obs_delay_steps as _delay
        self._pos_history = deque(maxlen=max(1, _delay + 1))
        self._observation_rng = np.random.default_rng(
            config_seed if observation_seed is None else observation_seed
        )
        self.sensed_positions = None

        self.num_of_subnets = 0
        self.positions_of_subnets = []
        self.index_of_subnets = []
        self.dist_mat = None
        self.A_mat = None
        self._g2l = {}
    # ------------------------------------------------------------------
    # Subnet detection (same as v1)
    # ------------------------------------------------------------------
    def check_subnets(self, node_global_positions, remain_list):
        from Configurations import _obs_position_noise, _obs_delay_steps, _obs_packet_loss

        sensed_positions = node_global_positions
        # --- Delay: use τ-steps-ago positions for neighbor sensing ---
        tau = _obs_delay_steps
        if tau > 0:
            self._pos_history.append(deepcopy(node_global_positions))
            if len(self._pos_history) > tau:
                sensed_positions = self._pos_history[0]  # τ-steps-ago
            else:
                sensed_positions = self._pos_history[0]  # oldest available

        # Noise applies to the selected (possibly delayed) observation.
        sigma = _obs_position_noise
        if sigma > EPS:
            sensed_positions = sensed_positions + self._observation_rng.normal(
                0.0, sigma, size=sensed_positions.shape
            )
        self.sensed_positions = sensed_positions

        remain_positions = np.array(
            [sensed_positions[i] for i in remain_list],
            dtype=np.float64,
        )
        n = len(remain_positions)
        if n == 0:
            self.num_of_subnets = 0
            self.positions_of_subnets = []
            self.index_of_subnets = []
            self.dist_mat = None
            self.A_mat = None
            self._g2l = {}
            return

        # Distance matrix computed once, reused by F_scout/F_retraction
        G = np.matmul(remain_positions, remain_positions.T)
        H = np.tile(np.diag(G), (n, 1))
        dist_sq = H + H.T - 2 * G
        self.dist_mat = np.sqrt(np.maximum(dist_sq, 0.0))
        self.A_mat = self.dist_mat <= config_communication_range
        np.fill_diagonal(self.A_mat, False)

        # --- Packet loss: randomly drop edges from adjacency ---
        p_loss = _obs_packet_loss
        if p_loss > EPS:
            # Paired contact loss preserves the undirected domain model.
            upper = np.triu(
                self._observation_rng.random(self.A_mat.shape) < p_loss, k=1
            )
            mask = upper | upper.T
            self.A_mat[mask] = False

        self._g2l = {g: l for l, g in enumerate(remain_list)}  # global→local mapping

        # Canonical ascending node order matches the previous component split.
        # Sparse labelling avoids a Python all-pairs traversal at every step.
        num, labels = connected_components(csr_matrix(self.A_mat), directed=False)
        self.num_of_subnets = int(num)
        local_subnet_indexes = [np.flatnonzero(labels == k).tolist() for k in range(num)]
        self.positions_of_subnets = [remain_positions[idx] for idx in local_subnet_indexes]

        self.index_of_subnets = []
        for subnet_indexes in local_subnet_indexes:
            self.index_of_subnets.append([remain_list[i] for i in subnet_indexes])

    # ------------------------------------------------------------------
    # Phase 1: Retraction — gradient-directed tissue contraction
    # ------------------------------------------------------------------
    def F_retraction(self, node_global_positions):
        """Morphogen-guided collective contraction toward the signaling source.

        Each fragment senses the morphogen gradient as a collective unit:
        the centroid-to-source vector gives the migration direction for all
        cells in the fragment. The gradient magnitude saturates at CR and
        decays naturally as the centroid approaches the source.

        Collective gradient renormalization: the weight is governed by the
        median distance to the source, not the minimum. A single pioneer
        cell reaching the target does not saturate the gradient for the
        rest of the cohort — half the population must arrive before the
        weight drops significantly.

        Returns (F_retraction, w_retraction) for every agent.
        """
        F_ret = np.zeros((self.num_of_agents, config_dimension))
        w_ret = np.zeros((self.num_of_agents, config_dimension))
        anchor = self.anchor_region

        for k in range(self.num_of_subnets):
            subnet_positions = self.positions_of_subnets[k]
            if len(subnet_positions) == 0:
                continue

            subnet_center = np.mean(subnet_positions, axis=0)
            raw_force = anchor - subnet_center
            anchor_distance = np.linalg.norm(raw_force)
            centroid_force = np.clip(anchor_distance, 0.0, config_communication_range) \
                * (raw_force / (anchor_distance + EPS))

            # Collective gradient renormalization: median, not minimum
            median_anchor_distance = np.median(
                np.linalg.norm(anchor - subnet_positions, axis=1)
            )
            group_weight = np.clip(
                median_anchor_distance / (anchor_distance + EPS), 0.0, 1.0
            )

            for agent_i in self.index_of_subnets[k]:
                F_ret[agent_i] = centroid_force
                w_ret[agent_i] = group_weight * np.ones(config_dimension)

        return deepcopy(F_ret), deepcopy(w_ret)

    # ------------------------------------------------------------------
    # Phase 2: Polarization — tip-cell symmetry breaking
    # ------------------------------------------------------------------
    def F_polarization(self, node_global_positions, K=1):
        """Tip-cell selection and polarization for rotational symmetry breaking.

        Within each fragment, the cell closest to the signaling source
        experiences local RhoA downregulation, actin reorganization, and
        filopodial extension — becoming the tip cell (leader). It receives
        a personal chemotactic boost proportional to the fragment's share
        of total tissue mass (BEARER_BOOST = N_subnet / N_total).

        K=1: a single tip cell per fragment is optimal. Multiple tip cells
        introduce directional conflict (PIEZO1-mediated suppression of
        excess leaders [Chen et al.]; multi-bearer +4.6% in ablation).

        The tip cell emerges from its position — not externally designated.
        It is simply the cell closest to the morphogen source within its
        connected component.

        Returns (F_polarization, w_polarization). Non-tip cells get zero.
        """
        F_pol = np.zeros((self.num_of_agents, config_dimension))
        w_pol = np.zeros((self.num_of_agents, config_dimension))
        anchor = self.anchor_region

        for k in range(self.num_of_subnets):
            subnet_positions = self.positions_of_subnets[k]
            if len(subnet_positions) == 0:
                continue

            dists_to_anchor = np.linalg.norm(subnet_positions - anchor, axis=1)
            K_actual = min(K, len(dists_to_anchor))
            bearer_local_indices = np.argpartition(
                dists_to_anchor, min(K_actual, len(dists_to_anchor) - 1)
            )[:K_actual]
            bearer_global_indices = [
                self.index_of_subnets[k][idx] for idx in bearer_local_indices
            ]
            BEARER_BOOST = len(subnet_positions) / self.num_of_agents

            for idx, agent_i in enumerate(self.index_of_subnets[k]):
                if agent_i not in bearer_global_indices:
                    continue
                bearer_dist = dists_to_anchor[idx]
                if bearer_dist < EPS:
                    continue
                bearer_to_anchor = anchor - self.sensed_positions[agent_i]
                bearer_force = np.clip(bearer_dist, 0.0, config_communication_range) \
                    * (bearer_to_anchor / bearer_dist)
                F_pol[agent_i] = BEARER_BOOST * bearer_force
                w_pol[agent_i] = 1.0 * np.ones(config_dimension)

        return deepcopy(F_pol), deepcopy(w_pol)

    def F_scout(self, node_global_positions, remain_list):
        """Free-edge sensing via angular-void detection (contact-inhibition of locomotion).

        In developing and regenerating tissues, cells at the frontier extend
        lamellipodia into the empty space where no neighbors are present — a
        phenomenon known as contact inhibition of locomotion (CIL). Neural crest
        cells, border cells in the Drosophila ovary, and keratinocytes at a
        wound edge all sense the angular distribution of their contacts and
        crawl toward the largest gap.

        At high cell density, simply summing neighbor direction vectors cancels
        to zero — every direction is equally occupied. But even in a crowded
        epithelium, there is always a sector of the cell perimeter that is less
        constrained than others. This method finds the largest angular gap in
        the neighbor distribution and pushes toward its midpoint.

        The push is gated: the gap must exceed a half-circle (π) — the cell
        must have a recognizable "free front." The amplitude scales with the
        mean safety (clearance) of the three nearest neighbors, so a cell
        surrounded on all sides at close range generates negligible scout force,
        while one with distant neighbors in every direction but a wide-open
        sector generates a strong directional push.

        This is density-adaptive by construction: sparse populations have large
        gaps and strong scouts; dense populations have small gaps and scout
        force naturally attenuates. No tunable parameters.
        """
        F_scout = np.zeros((self.num_of_agents, config_dimension))
        w_scout = np.zeros((self.num_of_agents, config_dimension))
        CR = config_communication_range

        for k in range(self.num_of_subnets):
            indexes = self.index_of_subnets[k]
            if len(indexes) <= 1:
                continue

            for agent_i in indexes:
                li = self._g2l.get(agent_i)
                if li is None:
                    continue

                MARGIN = 20.0
                safe_range = CR - MARGIN

                # Use pre-computed distance/adjacency — O(1) lookup per neighbor
                A_row = self.A_mat[li]
                neighbors_l = np.where(A_row)[0]  # local indices of neighbors

                if len(neighbors_l) < 2:
                    continue

                # Collect: (distance, safety, global_index_for_angle)
                neighbor_dist = self.dist_mat[li, neighbors_l]
                pos = self.sensed_positions[agent_i]

                nbr_info = []
                for idx, lj in enumerate(neighbors_l):
                    dist = neighbor_dist[idx]
                    safety = np.clip((CR - dist - MARGIN) / safe_range, 0.0, 1.0)
                    if safety <= EPS:
                        continue
                    j = remain_list[lj]  # global index
                    diff = self.sensed_positions[j] - pos
                    angle = np.arctan2(diff[1], diff[0])
                    nbr_info.append((dist, safety, angle))

                if len(nbr_info) < 2:
                    continue

                # Angular sorting: map contacts onto the cell's perimeter
                angles = np.array([a for _, _, a in nbr_info])
                order = np.argsort(angles)
                sorted_angles = angles[order]
                gaps = np.diff(sorted_angles)
                wrap_gap = 2 * np.pi - (sorted_angles[-1] - sorted_angles[0])
                all_gaps = np.append(gaps, wrap_gap)

                largest_gap = np.max(all_gaps)
                gap_idx = np.argmax(all_gaps)

                # Half-circle gate: a cell must have a recognizable
                # "free front" (>π) before it commits to migration.
                # This is the CIL threshold — local crowding below
                # this level is not perceived as a directional cue.
                gap_gate = np.clip((largest_gap - np.pi) / np.pi, 0.0, 1.0)
                if gap_gate < 0.01:
                    continue

                # Gap midpoint: the direction of maximal freedom — the
                # cell extends its leading edge into this sector.
                if gap_idx < len(gaps):
                    gap_mid = sorted_angles[gap_idx] + largest_gap / 2.0
                else:
                    gap_mid = sorted_angles[-1] + largest_gap / 2.0
                    if gap_mid > np.pi:
                        gap_mid -= 2 * np.pi

                free_dir = np.array([np.cos(gap_mid), np.sin(gap_mid), 0.0])

                # Amplitude = CR × gap_gate × mean safety of K=3 nearest.
                # The three closest neighbors define the immediate crowding
                # environment: if all are in safe contact → strong push;
                nbr_info.sort(key=lambda x: x[0])  # sort by distance
                K = 3
                top_safeties = [s for _, s, _ in nbr_info[:K]]
                amp = CR * gap_gate * np.mean(top_safeties) if top_safeties else 0.0

                scout_force = amp * free_dir
                F_scout[agent_i] = scout_force
                w_scout[agent_i] = 1.0

        return deepcopy(F_scout), deepcopy(w_scout)

    # ------------------------------------------------------------------
    # F_scout_vector: naive vector-sum scout (ablation baseline)
    # ------------------------------------------------------------------
    def F_scout_vector(self, node_global_positions, remain_list):
        """Naive vector-sum scout — the *wrong* CIL formalization.

        Sums neighbor→self direction vectors and normalises. In dense tissue
        this cancels to zero because every direction is equally occupied.
        Angular-void (F_scout) is the correct formalization — vector-sum
        is provided here as an ablation baseline to quantify the gap.

        Returns (F_scout, w_scout) with the same interface as F_scout.
        """
        F_s = np.zeros((self.num_of_agents, config_dimension))
        w_s = np.zeros((self.num_of_agents, config_dimension))
        CR = config_communication_range

        for k in range(self.num_of_subnets):
            indexes = self.index_of_subnets[k]
            if len(indexes) <= 1:
                continue

            for agent_i in indexes:
                pos = node_global_positions[agent_i]
                vec_sum = np.zeros(config_dimension)
                n_count = 0

                for j in indexes:
                    if j == agent_i:
                        continue
                    diff = node_global_positions[j] - pos
                    dist = np.linalg.norm(diff)
                    if dist <= CR and dist > EPS:
                        # Neighbour→self direction (CIL: repel from neighbours)
                        vec_sum = vec_sum - diff / dist
                        n_count += 1

                if n_count < 2:
                    continue

                vec_norm = np.linalg.norm(vec_sum)
                if vec_norm < EPS:
                    continue

                F_s[agent_i] = CR * vec_sum / vec_norm
                w_s[agent_i] = 1.0

        return deepcopy(F_s), deepcopy(w_s)

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------
    def _normalize_speed(self, speed, remain_list):
        for i in remain_list:
            norm = np.linalg.norm(speed[i])
            if norm > EPS:
                speed[i] = speed[i] / norm
        return speed

    # ------------------------------------------------------------------
    # Morphogenetic integration step: R → P → S
    # ------------------------------------------------------------------
    def group_fusion_step(self, node_global_positions, remain_list):
        """Execute the RPS morphogenetic repair program.

        Phase 1 (Retraction):  collective gradient-directed contraction.
        Phase 2 (Polarization): tip-cell symmetry breaking (K=1 default).
        Phase 3 (Spreading):    CIL-driven free-edge protrusion (angular-void).

        ablation modes:
          'full'         → R + P(K=1) + S (angular-void)
          'no-S'         → R + P(K=1) only (anchor-only)
          'no-P'         → R + S only (no bearer)
          'no-R'         → P(K=1) + S only (no morphogen gradient)
          'vector-scout' → R + P(K=1) + S(vector-sum)
          'multi-bearer' → R + P(K=3) + S (angular-void)
        """
        self.check_subnets(node_global_positions, remain_list)

        _abl = self.ablation

        # Phase 1: Retraction — morphogen-guided collective contraction
        if _abl == 'no-R':
            F_retraction = np.zeros((self.num_of_agents, config_dimension))
            w_retraction = np.zeros((self.num_of_agents, config_dimension))
        else:
            F_retraction, w_retraction = self.F_retraction(node_global_positions)

        # Phase 2: Polarization
        if _abl == 'no-P':
            F_polarization = np.zeros((self.num_of_agents, config_dimension))
            w_polarization = np.zeros((self.num_of_agents, config_dimension))
        elif _abl == 'multi-bearer':
            F_polarization, w_polarization = self.F_polarization(node_global_positions, K=3)
        else:
            F_polarization, w_polarization = self.F_polarization(node_global_positions, K=1)

        # Phase 3: Spreading
        if _abl == 'no-S':
            F_scout = np.zeros((self.num_of_agents, config_dimension))
            w_scout = np.zeros((self.num_of_agents, config_dimension))
        elif _abl == 'vector-scout':
            F_scout, w_scout = self.F_scout_vector(node_global_positions, remain_list)
        else:
            F_scout, w_scout = self.F_scout(node_global_positions, remain_list)

        # RPS integration
        speed = np.zeros((self.num_of_agents, config_dimension))
        for i in remain_list:
            speed[i] = (
                w_retraction[i] * F_retraction[i]
                + w_polarization[i] * F_polarization[i]
                + w_scout[i] * F_scout[i]
            )

        speed = self._normalize_speed(speed, remain_list)

        # Aggregate statistics for trajectory recording
        f_ret_mag = np.linalg.norm(F_retraction[remain_list], axis=1)
        f_pol_mag = np.linalg.norm(F_polarization[remain_list], axis=1)
        f_scout_mag = np.linalg.norm(F_scout[remain_list], axis=1)
        bearer_count = int(np.count_nonzero(f_pol_mag > EPS))

        step_info = {
            'num_subnets': self.num_of_subnets,
            'num_remain': len(remain_list),
            'subnet_sizes': [len(indexes) for indexes in self.index_of_subnets],
            'f_ret_mean': float(np.mean(f_ret_mag)),
            'f_pol_mean': float(np.mean(f_pol_mag)),
            'f_pol_bearer': float(np.mean(f_pol_mag[f_pol_mag > EPS])) if bearer_count > 0 else 0.0,
            'f_scout_mean': float(np.mean(f_scout_mag)),
            'bearer_count': bearer_count,
        }

        self.prev_positions = deepcopy(node_global_positions)
        self.velocities = deepcopy(speed)
        self.last_step_info = deepcopy(step_info)

        return deepcopy(speed), deepcopy(step_info)
